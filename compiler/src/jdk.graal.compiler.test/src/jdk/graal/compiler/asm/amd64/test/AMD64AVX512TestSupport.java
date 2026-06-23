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

import java.util.EnumSet;

import jdk.vm.ci.amd64.AMD64;
import jdk.vm.ci.amd64.AMD64.CPUFeature;
import jdk.vm.ci.code.TargetDescription;

/**
 * Shared helpers for the AMD64 AVX-512 high-register encoding tests.
 */
final class AMD64AVX512TestSupport {

    private AMD64AVX512TestSupport() {
    }

    /**
     * A synthetic target with full AVX-512 (plus F16C) so that the high vector registers
     * ({@code xmm16-31}) are addressable. The tests only encode instructions, so the target need not
     * match the host CPU.
     */
    static TargetDescription avx512Target() {
        EnumSet<CPUFeature> features = EnumSet.of(
                        CPUFeature.SSE, CPUFeature.SSE2, CPUFeature.AES, CPUFeature.AVX, CPUFeature.AVX2, CPUFeature.F16C,
                        CPUFeature.AVX512F, CPUFeature.AVX512BW, CPUFeature.AVX512VL, CPUFeature.AVX512DQ, CPUFeature.AVX512_VAES);
        return new TargetDescription(new AMD64(features), true, 16, 4096, true);
    }

    /** Formats assembled bytes as space-separated, uppercase, two-digit hex. */
    static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02X ", b & 0xFF));
        }
        return sb.toString().trim();
    }
}
