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

import static jdk.graal.compiler.asm.amd64.AVXKind.AVXSize.XMM;
import static jdk.graal.compiler.asm.amd64.AVXKind.AVXSize.YMM;
import static jdk.graal.compiler.asm.amd64.AVXKind.AVXSize.ZMM;

import java.util.function.Consumer;
import java.util.function.IntFunction;

import org.junit.Test;

import jdk.graal.compiler.asm.amd64.AMD64Address;
import jdk.graal.compiler.asm.amd64.AMD64Assembler;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexAESOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexMoveOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRMOp;
import jdk.graal.compiler.asm.amd64.AMD64Assembler.VexRVMOp;
import jdk.graal.compiler.asm.amd64.AVXKind.AVXSize;
import jdk.graal.compiler.core.common.Stride;
import jdk.graal.compiler.test.GraalTest;
import jdk.vm.ci.amd64.AMD64;
import jdk.vm.ci.code.Register;

/**
 * Verifies the VEX&rarr;EVEX auto-upgrade that lets VEX-encoded instructions address the AVX-512
 * high registers ({@code xmm16-31}). When an operand is a high register the assembler transparently
 * emits the EVEX-encoded {@code variant} of the op. This test asserts that the auto-upgraded
 * encoding is byte-for-byte identical to using the EVEX variant explicitly &mdash; in particular
 * that the EVEX compressed-displacement scaling factor ({@code disp8 * N}) is applied to a memory
 * operand. If the scaling factor is taken from the VEX op instead (which uses no scaling), a small
 * non-zero displacement is mis-encoded and the CPU reads from a wrong address.
 *
 * The test is host-independent: it builds a synthetic AVX-512 target and only encodes (never
 * executes) the instructions, so it runs on any CPU. Note that {@code ymm16}/{@code zmm16} are the
 * same JVMCI register as {@code xmm16}; the operand width is carried by {@link AVXSize}, so the
 * {@link #SIZES} dimension exercises the high register at all widths.
 */
public class AMD64AVX512HighRegisterEncodingTest extends GraalTest {

    /** The AVX-512 high register that triggers the VEX-&gt;EVEX upgrade (xmm16 / ymm16 / zmm16). */
    private static final Register HIGH = AMD64.xmm16;
    private static final AVXSize[] SIZES = {XMM, YMM, ZMM};
    /** Mix of multiples and non-multiples of every tested tuple scale (4, 16, 32, 64). */
    private static final int[] DISPLACEMENTS = {0, 4, 8, 16, 32, 64, 100, 0x80, 256, 1024};

    /**
     * Addressing modes that exercise the distinct ModRM/SIB branches of the operand encoder, all of
     * which must apply the same EVEX displacement scaling:
     * <ul>
     * <li>{@code [rax+d]} - plain base + displacement.</li>
     * <li>{@code [r13+d]} - rbp-style base; mod=00 is unavailable so a displacement is forced even
     * for {@code d == 0}.</li>
     * <li>{@code [rsp+d]} - rsp base requires a SIB byte (with a no-index encoding).</li>
     * <li>{@code [rsp+rcx*4+d]} - SIB with base and index.</li>
     * <li>{@code [r13+rdx*8+d]} - SIB with an rbp-style base, which forces a displacement.</li>
     * </ul>
     */
    private static final AddressMode[] ADDRESS_MODES = {
                    new AddressMode("[rax+%d]", d -> new AMD64Address(AMD64.rax, d)),
                    new AddressMode("[r13+%d]", d -> new AMD64Address(AMD64.r13, d)),
                    new AddressMode("[rsp+%d]", d -> new AMD64Address(AMD64.rsp, d)),
                    new AddressMode("[rsp+rcx*4+%d]", d -> new AMD64Address(AMD64.rsp, AMD64.rcx, Stride.S4, d)),
                    new AddressMode("[r13+rdx*8+%d]", d -> new AMD64Address(AMD64.r13, AMD64.rdx, Stride.S8, d)),
    };

    private static final class AddressMode {
        private final String label;
        private final IntFunction<AMD64Address> build;

        AddressMode(String label, IntFunction<AMD64Address> build) {
            this.label = label;
            this.build = build;
        }
    }

    private static String hexOf(Consumer<AMD64Assembler> emit) {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());
        emit.accept(asm);
        return AMD64AVX512TestSupport.hex(asm.copy(0, asm.position()));
    }

    /**
     * For every vector size, addressing mode, and displacement, an instruction that is auto-upgraded
     * to EVEX (because an operand is a high register) must encode exactly like the explicit EVEX
     * variant. Covers the three memory-operand families that the fix touches: register-memory (RM),
     * memory-register store (MR), and register-register-memory (RVM).
     */
    @Test
    public void autoUpgradeMatchesExplicitEvex() {
        for (AVXSize size : SIZES) {
            for (AddressMode mode : ADDRESS_MODES) {
                for (int disp : DISPLACEMENTS) {
                    AMD64Address addr = mode.build.apply(disp);
                    String where = String.format(mode.label, disp);

                    // RM load: dst <- [mem]. VSQRTPS uses the FVM tuple (N = 16/32/64).
                    assertEncodings("vsqrtps", size, where,
                                    a -> VexRMOp.EVSQRTPS.emit(a, size, HIGH, addr),
                                    a -> VexRMOp.VSQRTPS.emit(a, size, HIGH, addr));

                    // RM broadcast with a fixed-element tuple. VPBROADCASTD uses T1S_32BIT (N = 4).
                    assertEncodings("vpbroadcastd", size, where,
                                    a -> VexRMOp.EVPBROADCASTD.emit(a, size, HIGH, addr),
                                    a -> VexRMOp.VPBROADCASTD.emit(a, size, HIGH, addr));

                    // MR store: [mem] <- src. VMOVDQU32 uses the FVM tuple.
                    assertEncodings("vmovdqu32(store)", size, where,
                                    a -> VexMoveOp.EVMOVDQU32.emit(a, size, addr, HIGH),
                                    a -> VexMoveOp.VMOVDQU32.emit(a, size, addr, HIGH));

                    // RVM: dst <- op(src1, [mem]). VADDPS uses the FVM tuple.
                    assertEncodings("vaddps", size, where,
                                    a -> VexRVMOp.EVADDPS.emit(a, size, HIGH, AMD64.xmm1, addr),
                                    a -> VexRVMOp.VADDPS.emit(a, size, HIGH, AMD64.xmm1, addr));
                }
            }
        }
    }

    @Test
    public void explicitW0EvexVariantsAutoUpgrade() {
        assertEncodings("vpmaddubsw", YMM, "high registers",
                        a -> VexRVMOp.EVPMADDUBSW.emit(a, YMM, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> a.vpmaddubsw(AMD64.xmm16, AMD64.xmm17, AMD64.xmm18, YMM));
        assertEncodings("vpmaddwd", YMM, "high registers",
                        a -> VexRVMOp.EVPMADDWD.emit(a, YMM, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> a.vpmaddwd(AMD64.xmm16, AMD64.xmm17, AMD64.xmm18, YMM));
    }

    @Test
    public void explicitW1EvexVariantsAutoUpgrade() {
        assertEncodings("vpmuldq", YMM, "high registers",
                        a -> VexRVMOp.EVPMULDQ.emit(a, YMM, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> VexRVMOp.VPMULDQ.emit(a, YMM, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18));
    }

    @Test
    public void aesEvexVariantsAutoUpgrade() {
        assertEncodings("vaesenc", XMM, "high registers",
                        a -> VexAESOp.EVAESENC.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> VexAESOp.VAESENC.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18));
        assertEncodings("vaesenclast", XMM, "high registers",
                        a -> VexAESOp.EVAESENCLAST.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> VexAESOp.VAESENCLAST.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18));
        assertEncodings("vaesdec", XMM, "high registers",
                        a -> VexAESOp.EVAESDEC.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> VexAESOp.VAESDEC.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18));
        assertEncodings("vaesdeclast", XMM, "high registers",
                        a -> VexAESOp.EVAESDECLAST.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18),
                        a -> VexAESOp.VAESDECLAST.emit(a, AMD64.xmm16, AMD64.xmm17, AMD64.xmm18));
    }

    private void assertEncodings(String mnemonic, AVXSize size, String where, Consumer<AMD64Assembler> evex, Consumer<AMD64Assembler> vexAutoUpgrade) {
        assertDeepEquals(String.format("%s %s %s: auto-upgraded VEX must encode like explicit EVEX", mnemonic, size, where),
                        hexOf(evex), hexOf(vexAutoUpgrade));
    }

    /**
     * Documents the canonical failing case from the review: {@code vsqrtps zmm16, [rax+0x40]}.
     * With the EVEX FVM tuple (N = 64 for ZMM) the displacement {@code 0x40} must compress to a
     * {@code disp8} byte of {@code 0x40 / 64 == 0x01}. The buggy auto-upgrade path emitted
     * {@code 0x40} unscaled, which the CPU reads as {@code 0x40 * 64 == 4096}.
     */
    @Test
    public void vsqrtpsZmm16Disp64CompressesDisplacement() {
        AMD64Address addr = new AMD64Address(AMD64.rax, 0x40);
        String upgraded = hexOf(a -> VexRMOp.VSQRTPS.emit(a, ZMM, AMD64.xmm16, addr));
        String explicit = hexOf(a -> VexRMOp.EVSQRTPS.emit(a, ZMM, AMD64.xmm16, addr));

        // vsqrtps zmm16, [rax+0x40] encoded as EVEX.512.0F.W0 51 /r with FVM tuple (N = 64 for ZMM):
        //   62 E1 7C 48  EVEX prefix (R'/B for zmm16+rax, L'L=10 => 512-bit, map 0F, W0)
        //   51           opcode (SQRTPS)
        //   40           ModRM mod=01 (disp8) reg=000 (zmm16) rm=000 (rax)
        //   01           compressed disp8 == 0x40 / 64
        String expected = "62 E1 7C 48 51 40 01";
        assertDeepEquals("explicit EVEX encoding of vsqrtps zmm16,[rax+0x40]", expected, explicit);
        assertDeepEquals("auto-upgraded vsqrtps zmm16,[rax+0x40] must match explicit EVEX", expected, upgraded);
    }
}
