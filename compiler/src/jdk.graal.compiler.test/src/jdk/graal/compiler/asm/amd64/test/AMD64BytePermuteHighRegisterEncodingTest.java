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
import static org.junit.Assert.assertArrayEquals;

import java.util.Arrays;
import java.util.function.Consumer;

import org.junit.Test;

import jdk.graal.compiler.asm.amd64.AMD64Assembler;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRMOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRVMIOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRVMOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRVMROp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexShiftOp;
import jdk.graal.compiler.debug.GraalError;
import jdk.graal.compiler.test.GraalTest;
import jdk.vm.ci.amd64.AMD64;

/**
 * Regression test for the AVX-512 byte-vector rearrange lowering
 * ({@code AMD64VectorShuffle.PermuteOpWithTemps} for {@code BYTE}+{@code YMM} when
 * {@code AVX512_VBMI} is unavailable). That path previously emitted {@code VPERM2I128} and
 * {@code VPBLENDVB}, neither of which has an EVEX-encoded form, so they fail with "illegal operand"
 * when the register allocator assigns an AVX-512 high register ({@code xmm16-31}) - which it does
 * whenever AVX512F is present (e.g. Skylake-X / Cascade Lake, which have full AVX-512 but no VBMI).
 *
 * The fix re-expresses the YMM byte permute with EVEX-encodable instructions. This test verifies
 * (1) that the replacement instruction sequence encodes cleanly with high registers, and (2) that
 * the original instructions genuinely reject them - documenting why the EVEX path is required.
 *
 * The test is host-independent: it builds a synthetic AVX-512 target and only encodes (never
 * executes) the instructions, so it runs on any CPU.
 */
public class AMD64BytePermuteHighRegisterEncodingTest extends GraalTest {

    /**
     * The EVEX instruction sequence used by the fixed {@code emitBytePermute} (YMM) path must encode
     * without error when every vector operand is a high register. Before the fix this sequence used
     * VPERM2I128/VPBLENDVB and threw {@link GraalError}.
     */
    @Test
    public void evexBytePermuteSequenceEncodesWithHighRegisters() {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());

        // result / source / index / temporaries all in xmm16-31, opmask in k1, mirroring the
        // registers the allocator may hand to PermuteOpWithTemps under EVEX.
        var result = AMD64.xmm16;
        var source = AMD64.xmm17;
        var index = AMD64.xmm18;
        var xtmp1 = AMD64.xmm19;
        var xtmp2 = AMD64.xmm20;
        var xtmp3 = AMD64.xmm21;
        var ktmp = AMD64.k1;

        VexRVMIOp.EVSHUFI64X2.emit(asm, YMM, xtmp1, source, source, 0x00);
        VexRVMOp.EVPSHUFB.emit(asm, YMM, xtmp1, xtmp1, index);
        VexRVMIOp.EVSHUFI64X2.emit(asm, YMM, xtmp2, source, source, 0x03);
        VexRVMOp.EVPSHUFB.emit(asm, YMM, xtmp2, xtmp2, index);
        VexShiftOp.EVPSLLD.emit(asm, YMM, xtmp3, index, 3);
        VexRMOp.EVPMOVB2M.emit(asm, YMM, ktmp, xtmp3);
        VexRVMOp.EVPBLENDMB.emit(asm, YMM, result, xtmp1, xtmp2, ktmp);

        assertTrue("the EVEX byte-permute sequence must emit bytes", asm.position() > 0);
    }

    /**
     * The EVEX replacement must also preserve the semantics of a byte rearrange that selects from
     * both 128-bit halves. This is a host-independent model of the instruction sequence: it catches
     * the easy-to-make mistake of using VPERM2I128's {@code 0x11} immediate for VSHUFI64X2, whose
     * 256-bit form uses one selector bit per destination half.
     */
    @Test
    public void evexBytePermuteSequenceMatchesReferenceSemantics() {
        byte[] source = new byte[32];
        for (int i = 0; i < source.length; i++) {
            source[i] = (byte) (0x40 + i);
        }
        byte[] index = {
                        0, 1, 14, 15, 16, 17, 30, 31,
                        3, 19, 4, 20, 12, 28, 7, 23,
                        31, 30, 17, 16, 15, 14, 1, 0,
                        18, 2, 27, 11, 24, 8, 29, 13
        };

        byte[] expected = referenceBytePermute(source, index);
        assertArrayEquals(expected, evexBytePermute(source, index, 0x03));
        assertFalse("VSHUFI64X2 imm 0x11 would select [high, low], not broadcast the high half",
                        Arrays.equals(expected, evexBytePermute(source, index, 0x11)));
    }

    /**
     * Guards the reason the EVEX path exists: the VEX-only instructions it replaces cannot encode a
     * high register and must raise rather than emit a wrong instruction.
     */
    @Test
    public void vexOnlyOpsRejectHighRegisters() {
        assertRejectsHighRegister("VPERM2I128",
                        asm -> VexRVMIOp.VPERM2I128.emit(asm, YMM, AMD64.xmm16, AMD64.xmm1, AMD64.xmm1, 0x00));
        assertRejectsHighRegister("VPBLENDVB",
                        asm -> VexRVMROp.VPBLENDVB.emit(asm, YMM, AMD64.xmm16, AMD64.xmm1, AMD64.xmm2, AMD64.xmm3));
    }

    private void assertRejectsHighRegister(String name, Consumer<AMD64Assembler> emit) {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());
        try {
            emit.accept(asm);
            fail(name + " must reject an AVX-512 high register (it has no EVEX-encoded form)");
        } catch (GraalError expected) {
            // expected: "illegal operand xmm16"
        }
    }

    private static byte[] referenceBytePermute(byte[] source, byte[] index) {
        byte[] result = new byte[32];
        for (int i = 0; i < result.length; i++) {
            result[i] = source[index[i] & 0x1f];
        }
        return result;
    }

    private static byte[] evexBytePermute(byte[] source, byte[] index, int highHalfShuffleImm) {
        byte[] lowHalf = pshufb(vshufi64x2(source, 0x00), index);
        byte[] highHalf = pshufb(vshufi64x2(source, highHalfShuffleImm), index);
        byte[] result = new byte[32];
        for (int i = 0; i < result.length; i++) {
            result[i] = (index[i] & 0x10) == 0 ? lowHalf[i] : highHalf[i];
        }
        return result;
    }

    private static byte[] vshufi64x2(byte[] source, int imm8) {
        byte[] result = new byte[32];
        for (int lane = 0; lane < 2; lane++) {
            int selector = (imm8 >> lane) & 0x1;
            System.arraycopy(source, selector * 16, result, lane * 16, 16);
        }
        return result;
    }

    private static byte[] pshufb(byte[] source, byte[] index) {
        byte[] result = new byte[32];
        for (int laneBase = 0; laneBase < 32; laneBase += 16) {
            for (int i = 0; i < 16; i++) {
                int selector = index[laneBase + i] & 0xff;
                result[laneBase + i] = (selector & 0x80) == 0 ? source[laneBase + (selector & 0x0f)] : 0;
            }
        }
        return result;
    }
}
