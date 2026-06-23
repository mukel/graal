/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package jdk.graal.compiler.asm.amd64.test;

import static jdk.graal.compiler.asm.amd64.AVXKind.AVXSize.YMM;

import java.util.function.Consumer;

import org.junit.Test;

import jdk.graal.compiler.asm.amd64.AMD64Assembler;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexMRIOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRMIOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRMOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRVMOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexShiftImmOp;
import jdk.graal.compiler.test.GraalTest;
import jdk.vm.ci.amd64.AMD64;
import jdk.vm.ci.code.Register;

/**
 * Regression test guarding the VEX-&gt;EVEX auto-promotion that several AMD64 intrinsics rely on for
 * correctness under full AVX-512.
 *
 * A handful of ops (e.g. {@code AVXByteCompress}, {@code AMD64EncodeArrayOp},
 * {@code AMD64ArrayCopyWithConversionsOp}, {@code AMD64StringLatin1InflateOp},
 * {@code AMD64CountPositivesOp}) emit VEX-named SIMD opcodes on allocator-assigned vector registers
 * that, under full AVX-512, may be the high registers xmm16-31. Those ops are NOT register-pinned;
 * they are safe only because each opcode they use has a linked EVEX variant, so the assembler
 * automatically re-encodes the instruction as EVEX (which can address xmm16-31) when a high register
 * appears. If such a variant link were ever removed, the VEX-only form would instead raise an
 * "illegal operand" error at emit time - a silent correctness regression for those ops.
 *
 * This test pins that contract: for every such opcode it asserts the encoding uses the VEX prefix
 * (0xC4/0xC5) for low registers but auto-promotes to the EVEX prefix (0x62) for a high register. A
 * removed EVEX variant makes the high-register emit throw (or drop the 0x62 prefix), failing here.
 *
 * The test is host-independent: it builds a synthetic AVX-512 target and only encodes (never
 * executes) the instructions, so it runs on any CPU.
 */
public class AMD64AutoPromoteHighRegisterEncodingTest extends GraalTest {

    private static final int VEX_3BYTE = 0xC4;
    private static final int VEX_2BYTE = 0xC5;
    private static final int EVEX = 0x62;

    /** Low registers (xmm0-15) - addressable by VEX. */
    private static final Register LO0 = AMD64.xmm0;
    private static final Register LO1 = AMD64.xmm1;
    private static final Register LO2 = AMD64.xmm2;
    /** High registers (xmm16-31) - require EVEX. */
    private static final Register HI0 = AMD64.xmm16;
    private static final Register HI1 = AMD64.xmm17;
    private static final Register HI2 = AMD64.xmm18;

    @Test
    public void lowRegistersUseVex() {
        forEachOpcode(this::assertVexPrefix, LO0, LO1, LO2);
    }

    @Test
    public void highRegistersAutoPromoteToEvex() {
        forEachOpcode(this::assertEvexPrefix, HI0, HI1, HI2);
    }

    /**
     * Emits every auto-promotion-dependent opcode with the given (low or high) register triple and
     * runs {@code check} on each.
     */
    private void forEachOpcode(PrefixCheck check, Register r0, Register r1, Register r2) {
        check.accept("VPSHUFB", a -> VexRVMOp.VPSHUFB.emit(a, YMM, r0, r1, r2));
        check.accept("VPXOR", a -> VexRVMOp.VPXOR.emit(a, YMM, r0, r1, r2));
        check.accept("VPOR", a -> VexRVMOp.VPOR.emit(a, YMM, r0, r1, r2));
        check.accept("VPADDB", a -> VexRVMOp.VPADDB.emit(a, YMM, r0, r1, r2));
        // VPMADDUBSW/VPMADDWD had their high-register guards removed in favor of EVEX auto-promotion.
        check.accept("VPMADDUBSW", a -> VexRVMOp.VPMADDUBSW.emit(a, YMM, r0, r1, r2));
        check.accept("VPMADDWD", a -> VexRVMOp.VPMADDWD.emit(a, YMM, r0, r1, r2));
        check.accept("VPACKUSWB", a -> VexRVMOp.VPACKUSWB.emit(a, YMM, r0, r1, r2));
        check.accept("VPACKUSDW", a -> VexRVMOp.VPACKUSDW.emit(a, YMM, r0, r1, r2));
        check.accept("VPERMD", a -> VexRVMOp.VPERMD.emit(a, YMM, r0, r1, r2));
        check.accept("VPERMQ", a -> VexRMIOp.VPERMQ.emit(a, YMM, r0, r1, 0x1B));
        check.accept("VPMOVZXBW", a -> VexRMOp.VPMOVZXBW.emit(a, YMM, r0, r1));
        check.accept("VPBROADCASTD", a -> VexRMOp.VPBROADCASTD.emit(a, YMM, r0, r1));
        check.accept("VEXTRACTI128", a -> VexMRIOp.VEXTRACTI128.emit(a, YMM, r0, r1, 1));
        check.accept("VPSLLDQ", a -> VexShiftImmOp.VPSLLDQ.emit(a, YMM, r0, r1, 4));
    }

    private interface PrefixCheck {
        void accept(String name, Consumer<AMD64Assembler> emit);
    }

    private static byte[] emit(Consumer<AMD64Assembler> emit) {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());
        emit.accept(asm);
        return asm.copy(0, asm.position());
    }

    private void assertVexPrefix(String name, Consumer<AMD64Assembler> emitter) {
        byte[] bytes = emit(emitter);
        int prefix = bytes.length > 0 ? bytes[0] & 0xFF : -1;
        assertTrue(name + " should encode as VEX (0xC4/0xC5) for low registers, got " + AMD64AVX512TestSupport.hex(bytes),
                        prefix == VEX_3BYTE || prefix == VEX_2BYTE);
    }

    private void assertEvexPrefix(String name, Consumer<AMD64Assembler> emitter) {
        // Throws if the opcode has no EVEX variant to promote to (the regression we are guarding).
        byte[] bytes = emit(emitter);
        int prefix = bytes.length > 0 ? bytes[0] & 0xFF : -1;
        assertTrue(name + " must auto-promote to EVEX (0x62) for high registers xmm16-31, got " + AMD64AVX512TestSupport.hex(bytes),
                        prefix == EVEX);
    }
}
