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
package jdk.graal.compiler.core.amd64.test;

import java.util.function.IntUnaryOperator;

import org.junit.Test;

import jdk.graal.compiler.core.test.GraalCompilerTest;
import jdk.graal.compiler.test.AddModules;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * Correctness test for the 256-bit (YMM) cross-lane byte rearrange lowering in
 * {@code AMD64VectorShuffle} ({@code emitBytePermute}).
 *
 * On AVX-512 hardware WITHOUT VBMI (Skylake-X / Cascade Lake), {@code VPERMB} is unavailable and a
 * byte rearrange is synthesized from {@code VSHUFI64X2} (broadcast each 128-bit lane) +
 * {@code VPSHUFB} (within-lane shuffle) + an opmask blend. That fallback's lane-broadcast immediates
 * are subtle (VSHUFI64X2's selector encoding differs from VPERM2I128's), so this test exercises it
 * with shuffles that move bytes ACROSS the two 128-bit lanes - where a wrong broadcast immediate
 * would pull from the wrong half and corrupt the result.
 *
 * {@link GraalCompilerTest#test} compares the compiled result against interpreted execution, so the
 * test is correct on any host. On VBMI machines (e.g. the Zen 5 dev host) the rearrange lowers to a
 * single {@code VPERMB}; on no-VBMI AVX-512 CI machines it lowers to - and thus validates - the
 * {@code VSHUFI64X2}/{@code VPSHUFB}/blend fallback. On non-AVX-512 hosts it still validates the
 * corresponding 256-bit path.
 */
@AddModules("jdk.incubator.vector")
public class AMD64ByteRearrangeCorrectnessTest extends GraalCompilerTest {

    private static final VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_256;
    private static final int LEN = SPECIES.length();

    /**
     * Rearranges a 256-bit byte vector by a runtime shuffle (so the permute indices live in a vector
     * register, forcing the variable byte-permute lowering rather than a constant-table permute).
     */
    public static byte[] rearrangeSnippet(byte[] src, int[] indices) {
        byte[] dst = new byte[LEN];
        ByteVector v = ByteVector.fromArray(SPECIES, src, 0);
        VectorShuffle<Byte> shuffle = VectorShuffle.fromArray(SPECIES, indices, 0);
        v.rearrange(shuffle).intoArray(dst, 0);
        return dst;
    }

    private static byte[] sequentialBytes() {
        byte[] src = new byte[LEN];
        for (int i = 0; i < LEN; i++) {
            // Distinct, lane-distinguishable values so a wrong-half pull is observable.
            src[i] = (byte) (0x40 + i);
        }
        return src;
    }

    private void assertRearrange(int[] indices) {
        test("rearrangeSnippet", sequentialBytes(), indices);
    }

    /** Rearranges by the shuffle {@code out[i] = src[indexFunction(i)]}. */
    private void assertRearrange(IntUnaryOperator indexFunction) {
        int[] indices = new int[LEN];
        for (int i = 0; i < LEN; i++) {
            indices[i] = indexFunction.applyAsInt(i);
        }
        assertRearrange(indices);
    }

    @Test
    public void reverse() {
        // every byte crosses to the opposite 128-bit lane
        assertRearrange(i -> LEN - 1 - i);
    }

    @Test
    public void swapLanes() {
        // low half <-> high half
        assertRearrange(i -> i < LEN / 2 ? i + LEN / 2 : i - LEN / 2);
    }

    @Test
    public void interleaveAcrossLanes() {
        // each output byte alternates between the two source halves
        assertRearrange(i -> i % 2 == 0 ? i / 2 : LEN / 2 + i / 2);
    }

    @Test
    public void broadcastHighLaneByte() {
        // every output selects the top byte (high lane) - most sensitive to the high-lane immediate
        assertRearrange(i -> LEN - 1);
    }

    @Test
    public void rotateByOne() {
        assertRearrange(i -> (i + 1) % LEN);
    }

    @Test
    public void identity() {
        assertRearrange(i -> i);
    }
}
