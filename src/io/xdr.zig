// zxdrfile - XDR trajectory file library for Zig
//
// Provides readers and writers for GROMACS trajectory formats:
// - XTC: Compressed coordinate trajectories (read and write)
// - TRR: Full-precision trajectories (read and write)
//
// Based on the xdrfile implementation from mdtraj:
// https://github.com/mdtraj/mdtraj/tree/master/mdtraj/formats/xtc
//
// Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
// Copyright (c) 2014, Robert T. McGibbon (mdtraj modifications)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

pub const xtc = @import("xtc_format.zig");
pub const trr = @import("trr_format.zig");

// Re-export main types for convenience
pub const XtcReader = xtc.XtcReader;
pub const XtcWriter = xtc.XtcWriter;
pub const XtcFrame = xtc.XtcFrame;
pub const XtcError = xtc.XtcError;

pub const TrrReader = trr.TrrReader;
pub const TrrWriter = trr.TrrWriter;
pub const TrrFrame = trr.TrrFrame;
pub const TrrError = trr.TrrError;

test {
    @import("std").testing.refAllDecls(@This());
}
