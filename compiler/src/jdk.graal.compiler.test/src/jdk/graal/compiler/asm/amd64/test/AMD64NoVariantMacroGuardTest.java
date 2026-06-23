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
import jdk.graal.compiler.debug.GraalError;
import jdk.graal.compiler.test.GraalTest;
import jdk.vm.ci.amd64.AMD64;

/**
 * Verifies the defensive guards on the AMD64 SIMD assembler macros that have no EVEX-encoded variant
 * ({@code vptest}, {@code vpmovmskb}, {@code vpcmpeq*}, {@code vperm2f128}, {@code vperm2i128},
 * {@code vpblendd}). Because these cannot address the AVX-512 high registers ({@code xmm16-31}),
 * the macros raise an actionable {@link GraalError} when handed one (under full AVX-512) instead of
 * producing a misleading low-level encoding failure, and emit normally for {@code xmm0-15}.
 *
 * The test is host-independent: it builds a synthetic AVX-512 target and only encodes (never
 * executes) the instructions, so it runs on any CPU.
 */
public class AMD64NoVariantMacroGuardTest extends GraalTest {

    @Test
    public void macrosRejectHighRegisters() {
        assertRejects("vptest", a -> a.vptest(AMD64.xmm16, AMD64.xmm16, YMM));
        assertRejects("vpmovmskb", a -> a.vpmovmskb(AMD64.rax, AMD64.xmm16));
        assertRejects("vpcmpeqb", a -> a.vpcmpeqb(AMD64.xmm16, AMD64.xmm1, AMD64.xmm2));
        assertRejects("vpcmpeqd", a -> a.vpcmpeqd(AMD64.xmm1, AMD64.xmm16, AMD64.xmm2));
        assertRejects("vpcmpeqw", a -> a.vpcmpeqw(AMD64.xmm1, AMD64.xmm2, AMD64.xmm16));
        assertRejects("vperm2f128", a -> a.vperm2f128(AMD64.xmm16, AMD64.xmm1, AMD64.xmm2, 0x00));
        assertRejects("vperm2i128", a -> a.vperm2i128(AMD64.xmm16, AMD64.xmm1, AMD64.xmm2, 0x00));
        assertRejects("vpblendd", a -> a.vpblendd(AMD64.xmm16, AMD64.xmm1, AMD64.xmm2, 0x00, YMM));
    }

    @Test
    public void macrosEmitWithLowRegisters() {
        assertEmits("vptest", a -> a.vptest(AMD64.xmm0, AMD64.xmm1, YMM));
        assertEmits("vpmovmskb", a -> a.vpmovmskb(AMD64.rax, AMD64.xmm1));
        assertEmits("vpcmpeqb", a -> a.vpcmpeqb(AMD64.xmm0, AMD64.xmm1, AMD64.xmm2));
        assertEmits("vpmaddubsw", a -> a.vpmaddubsw(AMD64.xmm0, AMD64.xmm1, AMD64.xmm2, YMM));
        assertEmits("vpmaddwd", a -> a.vpmaddwd(AMD64.xmm0, AMD64.xmm1, AMD64.xmm2, YMM));
        assertEmits("vperm2i128", a -> a.vperm2i128(AMD64.xmm0, AMD64.xmm1, AMD64.xmm2, 0x00));
        assertEmits("vpblendd", a -> a.vpblendd(AMD64.xmm0, AMD64.xmm1, AMD64.xmm2, 0x00, YMM));
    }

    private void assertRejects(String name, Consumer<AMD64Assembler> emit) {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());
        try {
            emit.accept(asm);
            fail(name + " must reject an AVX-512 high register operand");
        } catch (GraalError expected) {
            assertTrue("error should name the instruction: " + expected.getMessage(),
                            expected.getMessage() != null && expected.getMessage().contains(name));
        }
    }

    private void assertEmits(String name, Consumer<AMD64Assembler> emit) {
        AMD64Assembler asm = new AMD64Assembler(AMD64AVX512TestSupport.avx512Target());
        emit.accept(asm);
        assertTrue(name + " must emit for xmm0-15", asm.position() > 0);
    }
}
