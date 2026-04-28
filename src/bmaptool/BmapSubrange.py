# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 tw=88 et ai si
#
# Copyright (c) 2026 Flipper FZCO
# License: GPLv2
# Author: Alexey Charkov <alchark@flipper.net>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 2,
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

"""
This module implements sparse-aware copying of a byte-range subrange from one
file to another, and provides the corresponding API through the
'BmapSubrange' class.

The copy preserves sparse holes: byte ranges that correspond to holes in the
source file are recreated as holes in the destination rather than being written
as explicit zeros.  This is different from ``dd conv=sparse``, which promotes
runs of zero bytes to holes regardless of whether the source had actual holes.
"""

import os
import errno
import logging

_log = logging.getLogger(__name__)


class Error(Exception):
    """A class for all exceptions raised by this module."""

    pass


# SEEK_DATA / SEEK_HOLE whence values for lseek(2).
_SEEK_DATA = 3
_SEEK_HOLE = 4

# Read/write chunk size (1 MiB).
_COPY_CHUNK = 1 << 20


class BmapSubrange:
    """Copy a byte-range subrange of a source file to a destination file.

    Only data regions of the source are written to the destination; source
    holes are skipped entirely, leaving whatever was already at those offsets
    in the destination untouched (or creating a sparse hole when the
    destination is newly created).

    Usage::

        obj = BmapSubrange(source_path, dest_path)
        obj.copy()

    All parameters except *source* and *dest* are optional.

    When the destination does not exist it is created and its final size is set
    to ``dest_seek + min(length, source_size - start)`` bytes.

    When the destination already exists it is opened without truncation.  An
    error is raised if its size is less than ``dest_seek + actual_length``.
    """

    def __init__(self, source, dest, start=0, length=None, dest_seek=0):
        """Initialise the copy object.

        :param source:    path to the source file.
        :param dest:      path to the destination file.
        :param start:     byte offset in the source file at which to begin
                          (default: 0).
        :param length:    maximum number of bytes to copy; ``None`` means copy
                          to the end of the source file (default: ``None``).
        :param dest_seek: byte offset in the destination file at which to
                          start writing (default: 0).
        :raises Error: if *source* cannot be stat'd or *start* is out of range.
        """
        try:
            src_size = os.stat(source).st_size
        except OSError as err:
            raise Error("cannot stat source file '%s': %s" % (source, err))

        if start > src_size:
            raise Error(
                "start offset %d is beyond the source file size %d"
                % (start, src_size)
            )

        if length is None:
            length = src_size - start

        self.source = source
        self.dest = dest
        self.start = start
        self.length = min(length, src_size - start)
        self.dest_seek = dest_seek

    def copy(self):
        """Perform the copy.

        If the destination does not exist it is created and sized to
        ``dest_seek + length`` bytes (trailing region becomes a sparse hole).
        If the destination already exists it is opened without truncation and
        an error is raised if it is too small to accommodate the write.

        Source data regions are written to the destination; source holes are
        skipped, leaving the corresponding destination bytes unchanged.

        :raises Error: on any I/O failure.
        """
        src_end = self.start + self.length
        dest_end = self.dest_seek + self.length

        try:
            src_fd = os.open(self.source, os.O_RDONLY)
        except OSError as err:
            raise Error("cannot open source file '%s': %s" % (self.source, err))

        try:
            _require_seekable(src_fd, self.source, is_dest=False)

            dest_exists = os.path.exists(self.dest)
            if dest_exists:
                dest_flags = os.O_WRONLY
                final_size = None  # do not alter destination size
            else:
                dest_flags = os.O_WRONLY | os.O_CREAT
                final_size = dest_end  # set size; leading/trailing gaps become holes

            dest_fd = os.open(self.dest, dest_flags, 0o666)
        except Error:
            os.close(src_fd)
            raise
        except OSError as err:
            os.close(src_fd)
            raise Error(
                "cannot open destination file '%s': %s" % (self.dest, err)
            )

        try:
            _require_seekable(dest_fd, self.dest, is_dest=True)

            if dest_exists:
                dest_size = os.fstat(dest_fd).st_size
                if dest_size < dest_end:
                    raise Error(
                        "destination '%s' is too small: its size is %d bytes "
                        "but writing %d bytes at offset %d requires at least "
                        "%d bytes"
                        % (self.dest, dest_size, self.length,
                           self.dest_seek, dest_end)
                    )

            _copy_sparse(
                src_fd, dest_fd, self.start, src_end,
                dest_start=self.dest_seek, final_size=final_size,
            )
        except OSError as err:
            raise Error("copy failed: %s" % err)
        finally:
            os.close(src_fd)
            os.close(dest_fd)


def _require_seekable(fd, path, is_dest):
    """Raise :exc:`Error` with a descriptive message if *fd* is not seekable.

    Non-seekable file descriptors (pipes, sockets, …) are incompatible with
    this module because:

    * The source requires ``SEEK_DATA`` / ``SEEK_HOLE`` to locate sparse
      regions and a backward seek to re-position before each read.
    * The destination requires ``pwrite`` at arbitrary offsets to recreate
      holes and ``ftruncate`` to set the final file size.
    """
    try:
        os.lseek(fd, 0, os.SEEK_CUR)
    except OSError as err:
        if err.errno != errno.ESPIPE:
            raise
        if is_dest:
            raise Error(
                "destination '%s' is not seekable (e.g. a pipe or socket); "
                "sparse-hole recreation requires a regular file as the "
                "destination" % path
            )
        else:
            raise Error(
                "source '%s' is not seekable (e.g. a pipe or socket); "
                "sparse-hole detection via SEEK_DATA/SEEK_HOLE requires a "
                "regular file as the source" % path
            )


def _copy_sparse(src_fd, dest_fd, src_start, src_end, dest_start=0, final_size=None):
    """Copy bytes [src_start, src_end) from *src_fd* to *dest_fd*.

    Data is written to *dest_fd* starting at byte offset *dest_start*.
    Source holes are skipped; the corresponding destination bytes are not
    written and remain unchanged.

    If *final_size* is not ``None``, ``ftruncate(dest_fd, final_size)`` is
    called after the loop, setting the destination to exactly that size.  Pass
    ``None`` when writing into an existing file that must not be resized.

    Falls back to copying all bytes as data when SEEK_HOLE / SEEK_DATA are not
    supported by the underlying file system.
    """
    src_pos = src_start
    dest_pos = dest_start
    seek_supported = True

    while src_pos < src_end:
        if seek_supported:
            # Locate the next data byte at or after src_pos.
            try:
                next_data = os.lseek(src_fd, src_pos, _SEEK_DATA)
            except OSError as err:
                if err.errno == errno.ENXIO:
                    # No more data from src_pos to EOF – the rest is a hole.
                    break
                if err.errno == errno.EINVAL:
                    # SEEK_DATA not supported; copy everything as plain data.
                    seek_supported = False
                    continue
                raise

            if next_data >= src_end:
                # All remaining bytes in our range are a hole.
                break

            # Advance dest_pos over the hole region (if any) – the gap in the
            # destination will become a sparse hole because we never write there.
            dest_pos += next_data - src_pos
            src_pos = next_data

            # Find where the data region ends (start of the next hole).
            try:
                next_hole = os.lseek(src_fd, src_pos, _SEEK_HOLE)
            except OSError as err:
                if err.errno == errno.ENXIO:
                    next_hole = src_end
                else:
                    raise

            data_end = min(next_hole, src_end)
        else:
            # SEEK_HOLE / SEEK_DATA unavailable – treat everything as data.
            data_end = src_end

        # Copy [src_pos, data_end) from the source to dest_pos in the destination.
        os.lseek(src_fd, src_pos, os.SEEK_SET)
        remaining = data_end - src_pos

        while remaining > 0:
            chunk = os.read(src_fd, min(remaining, _COPY_CHUNK))
            if not chunk:
                break
            view = memoryview(chunk)
            written = 0
            while written < len(chunk):
                n = os.pwrite(dest_fd, view[written:], dest_pos + written)
                written += n
            dest_pos += written
            remaining -= written
            if not seek_supported:
                src_pos += written

        if seek_supported:
            src_pos = data_end

    if final_size is not None:
        # Set the destination to the intended size.  Any gap between the last
        # written byte and final_size becomes a sparse hole.
        os.ftruncate(dest_fd, final_size)
    os.fsync(dest_fd)
