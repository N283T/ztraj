//! Element module for atomic number, symbol, and physical property handling.
//!
//! This module provides the Element enum and utility functions for converting
//! between element symbols and atomic numbers, following gemmi's approach.
//!
//! Reference: gemmi/include/gemmi/elem.hpp

const std = @import("std");

/// Element enum where the value equals the atomic number.
/// X (unknown) = 0, H = 1, He = 2, ..., Og = 118
pub const Element = enum(u8) {
    X = 0, // Unknown element
    H = 1,
    He = 2,
    Li = 3,
    Be = 4,
    B = 5,
    C = 6,
    N = 7,
    O = 8,
    F = 9,
    Ne = 10,
    Na = 11,
    Mg = 12,
    Al = 13,
    Si = 14,
    P = 15,
    S = 16,
    Cl = 17,
    Ar = 18,
    K = 19,
    Ca = 20,
    Sc = 21,
    Ti = 22,
    V = 23,
    Cr = 24,
    Mn = 25,
    Fe = 26,
    Co = 27,
    Ni = 28,
    Cu = 29,
    Zn = 30,
    Ga = 31,
    Ge = 32,
    As = 33,
    Se = 34,
    Br = 35,
    Kr = 36,
    Rb = 37,
    Sr = 38,
    Y = 39,
    Zr = 40,
    Nb = 41,
    Mo = 42,
    Tc = 43,
    Ru = 44,
    Rh = 45,
    Pd = 46,
    Ag = 47,
    Cd = 48,
    In = 49,
    Sn = 50,
    Sb = 51,
    Te = 52,
    I = 53,
    Xe = 54,
    Cs = 55,
    Ba = 56,
    La = 57,
    Ce = 58,
    Pr = 59,
    Nd = 60,
    Pm = 61,
    Sm = 62,
    Eu = 63,
    Gd = 64,
    Tb = 65,
    Dy = 66,
    Ho = 67,
    Er = 68,
    Tm = 69,
    Yb = 70,
    Lu = 71,
    Hf = 72,
    Ta = 73,
    W = 74,
    Re = 75,
    Os = 76,
    Ir = 77,
    Pt = 78,
    Au = 79,
    Hg = 80,
    Tl = 81,
    Pb = 82,
    Bi = 83,
    Po = 84,
    At = 85,
    Rn = 86,
    Fr = 87,
    Ra = 88,
    Ac = 89,
    Th = 90,
    Pa = 91,
    U = 92,
    Np = 93,
    Pu = 94,
    Am = 95,
    Cm = 96,
    Bk = 97,
    Cf = 98,
    Es = 99,
    Fm = 100,
    Md = 101,
    No = 102,
    Lr = 103,
    Rf = 104,
    Db = 105,
    Sg = 106,
    Bh = 107,
    Hs = 108,
    Mt = 109,
    Ds = 110,
    Rg = 111,
    Cn = 112,
    Nh = 113,
    Fl = 114,
    Mc = 115,
    Lv = 116,
    Ts = 117,
    Og = 118,

    /// Get the atomic number (same as enum value)
    pub fn atomicNumber(self: Element) u8 {
        return @intFromEnum(self);
    }

    /// Get the element symbol as a string
    pub fn symbol(self: Element) []const u8 {
        return element_names[@intFromEnum(self)];
    }

    /// Get van der Waals radius in Angstroms (for fallback radius guessing)
    /// Values from gemmi/elem.hpp, originally from Wikipedia/cctbx
    pub fn vdwRadius(self: Element) f64 {
        return vdw_radii[@intFromEnum(self)];
    }

    /// Get atomic mass in daltons (unified atomic mass units, u).
    /// Values are standard atomic weights from IUPAC 2021.
    /// Returns 0.0 for unknown (X) and for elements without a well-defined
    /// standard atomic weight (radioactive elements use most stable isotope).
    pub fn mass(self: Element) f64 {
        return atomic_masses[@intFromEnum(self)];
    }

    /// Check if this is a common biological element
    pub fn isBiological(self: Element) bool {
        return switch (self) {
            .H, .C, .N, .O, .P, .S => true,
            .Na, .Mg, .K, .Ca, .Mn, .Fe, .Co, .Ni, .Cu, .Zn => true,
            .Se, .I => true,
            else => false,
        };
    }
};

/// Parse element symbol string to Element.
/// Handles case-insensitive matching (e.g., "C", "c", "CA", "Ca", "ca" all work for Ca/C).
/// Returns Element.X for unknown/invalid symbols.
pub fn fromSymbol(symbol_str: []const u8) Element {
    if (symbol_str.len == 0) return .X;

    // Trim leading/trailing whitespace
    var start: usize = 0;
    var end: usize = symbol_str.len;
    while (start < end and symbol_str[start] == ' ') start += 1;
    while (end > start and symbol_str[end - 1] == ' ') end -= 1;

    const trimmed = symbol_str[start..end];
    if (trimmed.len == 0) return .X;

    // Convert to uppercase for comparison
    var upper: [2]u8 = .{ 0, 0 };
    upper[0] = std.ascii.toUpper(trimmed[0]);
    if (trimmed.len > 1 and std.ascii.isAlphabetic(trimmed[1])) {
        upper[1] = std.ascii.toUpper(trimmed[1]);
    }

    // Single letter elements (most common in PDB)
    if (upper[1] == 0) {
        return singleLetterElement(upper[0]);
    }

    // Two letter elements - search through the table
    for (element_names_upper, 0..) |name, i| {
        if (name.len == 2 and name[0] == upper[0] and name[1] == upper[1]) {
            return @enumFromInt(i);
        }
    }

    // Try single letter if two-letter lookup failed
    // (e.g., "CA" in atom name context might mean Carbon-alpha, not Calcium)
    return singleLetterElement(upper[0]);
}

/// Helper function for single-letter element lookup
fn singleLetterElement(c: u8) Element {
    return switch (c) {
        'H' => .H,
        'B' => .B,
        'C' => .C,
        'N' => .N,
        'O' => .O,
        'F' => .F,
        'P' => .P,
        'S' => .S,
        'K' => .K,
        'V' => .V,
        'Y' => .Y,
        'I' => .I,
        'W' => .W,
        'U' => .U,
        else => .X,
    };
}

/// Create Element from atomic number
pub fn fromAtomicNumber(n: u8) Element {
    if (n > 118) return .X;
    return @enumFromInt(n);
}

// Element names (mixed case, standard notation)
const element_names = [_][]const u8{
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
    "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};

// Element names uppercase (for case-insensitive lookup)
const element_names_upper = [_][]const u8{
    "X",  "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
    "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA", "SC",
    "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
    "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR", "NB", "MO", "TC",
    "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I",  "XE",
    "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",  "RE", "OS",
    "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR",
    "RA", "AC", "TH", "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF",
    "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT",
    "DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
};

// Van der Waals radii in Angstroms (from gemmi/elem.hpp)
const vdw_radii = [_]f64{
    // X     H     He    Li    Be    B     C     N     O     F     Ne
    1.00, 1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,
    // Na    Mg    Al    Si    P     S     Cl    Ar    K     Ca    Sc
    2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31, 2.11,
    // Ti    V     Cr    Mn    Fe    Co    Ni    Cu    Zn    Ga    Ge
    1.95, 1.06, 1.13, 1.19, 1.26, 1.13, 1.63, 1.40, 1.39, 1.87, 2.11,
    // As    Se    Br    Kr    Rb    Sr    Y     Zr    Nb    Mo    Tc
    1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 1.61, 1.42, 1.33, 1.75, 2.00,
    // Ru    Rh    Pd    Ag    Cd    In    Sn    Sb    Te    I     Xe
    1.20, 1.22, 1.63, 1.72, 1.58, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16,
    // Cs    Ba    La    Ce    Pr    Nd    Pm    Sm    Eu    Gd    Tb
    3.43, 2.68, 1.83, 1.86, 1.62, 1.79, 1.76, 1.74, 1.96, 1.69, 1.66,
    // Dy    Ho    Er    Tm    Yb    Lu    Hf    Ta    W     Re    Os
    1.63, 1.61, 1.59, 1.57, 1.54, 1.53, 1.40, 1.22, 1.26, 1.30, 1.58,
    // Ir    Pt    Au    Hg    Tl    Pb    Bi    Po    At    Rn    Fr
    1.22, 1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48,
    // Ra    Ac    Th    Pa    U     Np    Pu    Am    Cm    Bk    Cf
    2.83, 2.12, 1.84, 1.60, 1.86, 1.71, 1.67, 1.66, 1.65, 1.64, 1.63,
    // Es    Fm    Md    No    Lr    Rf    Db    Sg    Bh    Hs    Mt
    1.62, 1.61, 1.60, 1.59, 1.58, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    // Ds    Rg    Cn    Nh    Fl    Mc    Lv    Ts    Og
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
};

// Atomic masses in daltons (standard atomic weights, IUPAC 2021).
// For radioactive elements with no stable isotope, the mass of the most
// stable isotope is used. Returns 0.0 for X (unknown).
const atomic_masses = [_]f64{
    // X       H       He      Li      Be      B       C       N       O       F       Ne
    0.000,   1.008,   4.003,   6.941,   9.012,   10.811,  12.011,  14.007,  15.999,  18.998,  20.180,
    // Na      Mg      Al      Si      P       S       Cl      Ar      K       Ca      Sc
    22.990,  24.305,  26.982,  28.086,  30.974,  32.060,  35.450,  39.948,  39.098,  40.078,  44.956,
    // Ti      V       Cr      Mn      Fe      Co      Ni      Cu      Zn      Ga      Ge
    47.867,  50.942,  51.996,  54.938,  55.845,  58.933,  58.693,  63.546,  65.380,  69.723,  72.630,
    // As      Se      Br      Kr      Rb      Sr      Y       Zr      Nb      Mo      Tc
    74.922,  78.971,  79.904,  83.798,  85.468,  87.620,  88.906,  91.224,  92.906,  95.950,  97.000,
    // Ru      Rh      Pd      Ag      Cd      In      Sn      Sb      Te      I       Xe
    101.070, 102.906, 106.420, 107.868, 112.411, 114.818, 118.710, 121.760, 127.600, 126.904, 131.293,
    // Cs      Ba      La      Ce      Pr      Nd      Pm      Sm      Eu      Gd      Tb
    132.905, 137.327, 138.905, 140.116, 140.908, 144.242, 145.000, 150.360, 151.964, 157.250, 158.925,
    // Dy      Ho      Er      Tm      Yb      Lu      Hf      Ta      W       Re      Os
    162.500, 164.930, 167.259, 168.934, 173.045, 174.967, 178.490, 180.948, 183.840, 186.207, 190.230,
    // Ir      Pt      Au      Hg      Tl      Pb      Bi      Po      At      Rn      Fr
    192.217, 195.084, 196.967, 200.592, 204.383, 207.200, 208.980, 209.000, 210.000, 222.000, 223.000,
    // Ra      Ac      Th      Pa      U       Np      Pu      Am      Cm      Bk      Cf
    226.000, 227.000, 232.038, 231.036, 238.029, 237.000, 244.000, 243.000, 247.000, 247.000, 251.000,
    // Es      Fm      Md      No      Lr      Rf      Db      Sg      Bh      Hs      Mt
    252.000, 257.000, 258.000, 259.000, 266.000, 267.000, 268.000, 269.000, 270.000, 269.000, 278.000,
    // Ds      Rg      Cn      Nh      Fl      Mc      Lv      Ts      Og
    281.000, 282.000, 285.000, 286.000, 289.000, 290.000, 293.000, 294.000, 294.000,
};

// ============================================================================
// Tests
// ============================================================================

test "Element atomic number" {
    try std.testing.expectEqual(@as(u8, 0), Element.X.atomicNumber());
    try std.testing.expectEqual(@as(u8, 1), Element.H.atomicNumber());
    try std.testing.expectEqual(@as(u8, 6), Element.C.atomicNumber());
    try std.testing.expectEqual(@as(u8, 7), Element.N.atomicNumber());
    try std.testing.expectEqual(@as(u8, 8), Element.O.atomicNumber());
    try std.testing.expectEqual(@as(u8, 20), Element.Ca.atomicNumber());
    try std.testing.expectEqual(@as(u8, 26), Element.Fe.atomicNumber());
    try std.testing.expectEqual(@as(u8, 118), Element.Og.atomicNumber());
}

test "Element symbol" {
    try std.testing.expectEqualStrings("X", Element.X.symbol());
    try std.testing.expectEqualStrings("H", Element.H.symbol());
    try std.testing.expectEqualStrings("C", Element.C.symbol());
    try std.testing.expectEqualStrings("Ca", Element.Ca.symbol());
    try std.testing.expectEqualStrings("Fe", Element.Fe.symbol());
    try std.testing.expectEqualStrings("Og", Element.Og.symbol());
}

test "fromSymbol single letter" {
    try std.testing.expectEqual(Element.H, fromSymbol("H"));
    try std.testing.expectEqual(Element.C, fromSymbol("C"));
    try std.testing.expectEqual(Element.N, fromSymbol("N"));
    try std.testing.expectEqual(Element.O, fromSymbol("O"));
    try std.testing.expectEqual(Element.S, fromSymbol("S"));
    try std.testing.expectEqual(Element.P, fromSymbol("P"));
}

test "fromSymbol case insensitive" {
    try std.testing.expectEqual(Element.H, fromSymbol("h"));
    try std.testing.expectEqual(Element.C, fromSymbol("c"));
    try std.testing.expectEqual(Element.Fe, fromSymbol("FE"));
    try std.testing.expectEqual(Element.Fe, fromSymbol("fe"));
    try std.testing.expectEqual(Element.Fe, fromSymbol("Fe"));
    try std.testing.expectEqual(Element.Ca, fromSymbol("CA"));
    try std.testing.expectEqual(Element.Ca, fromSymbol("ca"));
}

test "fromSymbol two letter" {
    try std.testing.expectEqual(Element.He, fromSymbol("He"));
    try std.testing.expectEqual(Element.Li, fromSymbol("Li"));
    try std.testing.expectEqual(Element.Na, fromSymbol("Na"));
    try std.testing.expectEqual(Element.Mg, fromSymbol("Mg"));
    try std.testing.expectEqual(Element.Ca, fromSymbol("Ca"));
    try std.testing.expectEqual(Element.Fe, fromSymbol("Fe"));
    try std.testing.expectEqual(Element.Zn, fromSymbol("Zn"));
    try std.testing.expectEqual(Element.Se, fromSymbol("Se"));
    try std.testing.expectEqual(Element.Br, fromSymbol("Br"));
}

test "fromSymbol with whitespace" {
    try std.testing.expectEqual(Element.C, fromSymbol(" C"));
    try std.testing.expectEqual(Element.C, fromSymbol("C "));
    try std.testing.expectEqual(Element.C, fromSymbol(" C "));
    try std.testing.expectEqual(Element.Fe, fromSymbol(" Fe "));
}

test "fromSymbol unknown" {
    try std.testing.expectEqual(Element.X, fromSymbol(""));
    try std.testing.expectEqual(Element.X, fromSymbol("   "));
    try std.testing.expectEqual(Element.X, fromSymbol("Xx"));
    try std.testing.expectEqual(Element.X, fromSymbol("??"));
}

test "fromAtomicNumber" {
    try std.testing.expectEqual(Element.X, fromAtomicNumber(0));
    try std.testing.expectEqual(Element.H, fromAtomicNumber(1));
    try std.testing.expectEqual(Element.C, fromAtomicNumber(6));
    try std.testing.expectEqual(Element.Fe, fromAtomicNumber(26));
    try std.testing.expectEqual(Element.Og, fromAtomicNumber(118));
    try std.testing.expectEqual(Element.X, fromAtomicNumber(119));
    try std.testing.expectEqual(Element.X, fromAtomicNumber(255));
}

test "vdwRadius" {
    try std.testing.expectApproxEqAbs(@as(f64, 1.20), Element.H.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.70), Element.C.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.55), Element.N.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.52), Element.O.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.80), Element.S.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.31), Element.Ca.vdwRadius(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.26), Element.Fe.vdwRadius(), 0.01);
}

test "isBiological" {
    try std.testing.expect(Element.H.isBiological());
    try std.testing.expect(Element.C.isBiological());
    try std.testing.expect(Element.N.isBiological());
    try std.testing.expect(Element.O.isBiological());
    try std.testing.expect(Element.P.isBiological());
    try std.testing.expect(Element.S.isBiological());
    try std.testing.expect(Element.Fe.isBiological());
    try std.testing.expect(Element.Zn.isBiological());
    try std.testing.expect(!Element.X.isBiological());
    try std.testing.expect(!Element.Au.isBiological());
    try std.testing.expect(!Element.Og.isBiological());
}

test "mass biological elements" {
    // All values within 0.001 of IUPAC 2021 standard atomic weights
    try std.testing.expectApproxEqAbs(@as(f64, 0.000), Element.X.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.008), Element.H.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 12.011), Element.C.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 14.007), Element.N.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 15.999), Element.O.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 30.974), Element.P.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 32.060), Element.S.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 78.971), Element.Se.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 55.845), Element.Fe.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 65.380), Element.Zn.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 40.078), Element.Ca.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 24.305), Element.Mg.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 22.990), Element.Na.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 35.450), Element.Cl.mass(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 39.098), Element.K.mass(), 0.001);
}

test "mass lookup table length" {
    // Verify the table covers all 119 entries (X through Og)
    try std.testing.expectEqual(@as(usize, 119), atomic_masses.len);
}

test "mass positive for all known elements" {
    // Every element except X should have mass > 0
    for (1..119) |i| {
        const elem: Element = @enumFromInt(i);
        try std.testing.expect(elem.mass() > 0.0);
    }
    try std.testing.expectEqual(@as(f64, 0.0), Element.X.mass());
}
