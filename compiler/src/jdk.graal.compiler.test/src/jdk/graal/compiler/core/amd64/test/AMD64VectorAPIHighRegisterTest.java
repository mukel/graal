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

import static jdk.graal.compiler.core.common.cfg.AbstractControlFlowGraph.INVALID_BLOCK_ID;
import static org.junit.Assume.assumeTrue;

import org.junit.Test;

import jdk.graal.compiler.asm.amd64.AMD64BaseAssembler;
import jdk.graal.compiler.core.common.cfg.BasicBlock;
import jdk.graal.compiler.core.common.type.IntegerStamp;
import jdk.graal.compiler.core.common.type.StampFactory;
import jdk.graal.compiler.core.test.GraalCompilerTest;
import jdk.graal.compiler.lir.LIRInstruction;
import jdk.graal.compiler.lir.LIRValueUtil;
import jdk.graal.compiler.lir.StandardOp.ValueMoveOp;
import jdk.graal.compiler.lir.gen.LIRGenerationResult;
import jdk.graal.compiler.lir.phases.FinalCodeAnalysisPhase.FinalCodeAnalysisContext;
import jdk.graal.compiler.lir.phases.LIRPhase;
import jdk.graal.compiler.lir.phases.LIRSuites;
import jdk.graal.compiler.options.OptionValues;
import jdk.graal.compiler.test.AddModules;
import jdk.graal.compiler.vector.architecture.VectorArchitecture;
import jdk.graal.compiler.vector.architecture.VectorLoweringProvider;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.vm.ci.amd64.AMD64;
import jdk.vm.ci.amd64.AMD64Kind;
import jdk.vm.ci.code.TargetDescription;
import jdk.vm.ci.code.ValueUtil;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.PlatformKind;
import jdk.vm.ci.meta.Value;

/**
 * Regression test for GR-13757: on AVX-512 the register allocator must use the high vector registers
 * ({@code xmm16-31}) so that register-heavy Vector API kernels do not spill. The kernel below keeps
 * {@value #ACCUMULATORS} SIMD accumulators live across a loop; with only {@code xmm0-15} available
 * that exceeds the register file and forces spills, whereas with all 32 registers it fits.
 *
 * A final-code-analysis LIR phase (the pattern used by graal-enterprise's AVX-512 LIR tests)
 * inspects the post-allocation LIR of the compiled kernel and the test asserts that it uses at least
 * one high register and emits no SIMD spill moves.
 *
 * Requires full AVX-512 (the only configuration in which {@code xmm16-31} are allocatable); the test
 * is skipped otherwise.
 */
@AddModules("jdk.incubator.vector")
public class AMD64VectorAPIHighRegisterTest extends GraalCompilerTest {

    static final int ACCUMULATORS = 20;

    private boolean sawSimd;
    private boolean usesHighRegister;
    private int simdSpillMoves;

    /**
     * Sums {@link #ACCUMULATORS} strided sub-vectors of {@code in} into separate accumulators,
     * keeping all of them live across the loop, then writes them out. This mirrors the
     * register-blocked kernels (e.g. GEMM) that motivated unlocking xmm16-31.
     */
    public static void manyAccumulators(float[] in, float[] out, int iters) {
        VectorSpecies<Float> sp = FloatVector.SPECIES_PREFERRED;
        int l = sp.length();
        FloatVector a0 = FloatVector.fromArray(sp, in, 0 * l);
        FloatVector a1 = FloatVector.fromArray(sp, in, 1 * l);
        FloatVector a2 = FloatVector.fromArray(sp, in, 2 * l);
        FloatVector a3 = FloatVector.fromArray(sp, in, 3 * l);
        FloatVector a4 = FloatVector.fromArray(sp, in, 4 * l);
        FloatVector a5 = FloatVector.fromArray(sp, in, 5 * l);
        FloatVector a6 = FloatVector.fromArray(sp, in, 6 * l);
        FloatVector a7 = FloatVector.fromArray(sp, in, 7 * l);
        FloatVector a8 = FloatVector.fromArray(sp, in, 8 * l);
        FloatVector a9 = FloatVector.fromArray(sp, in, 9 * l);
        FloatVector a10 = FloatVector.fromArray(sp, in, 10 * l);
        FloatVector a11 = FloatVector.fromArray(sp, in, 11 * l);
        FloatVector a12 = FloatVector.fromArray(sp, in, 12 * l);
        FloatVector a13 = FloatVector.fromArray(sp, in, 13 * l);
        FloatVector a14 = FloatVector.fromArray(sp, in, 14 * l);
        FloatVector a15 = FloatVector.fromArray(sp, in, 15 * l);
        FloatVector a16 = FloatVector.fromArray(sp, in, 16 * l);
        FloatVector a17 = FloatVector.fromArray(sp, in, 17 * l);
        FloatVector a18 = FloatVector.fromArray(sp, in, 18 * l);
        FloatVector a19 = FloatVector.fromArray(sp, in, 19 * l);
        for (int i = 1; i < iters; i++) {
            int base = i * (ACCUMULATORS * l);
            a0 = FloatVector.fromArray(sp, in, base + 0 * l).add(a0);
            a1 = FloatVector.fromArray(sp, in, base + 1 * l).add(a1);
            a2 = FloatVector.fromArray(sp, in, base + 2 * l).add(a2);
            a3 = FloatVector.fromArray(sp, in, base + 3 * l).add(a3);
            a4 = FloatVector.fromArray(sp, in, base + 4 * l).add(a4);
            a5 = FloatVector.fromArray(sp, in, base + 5 * l).add(a5);
            a6 = FloatVector.fromArray(sp, in, base + 6 * l).add(a6);
            a7 = FloatVector.fromArray(sp, in, base + 7 * l).add(a7);
            a8 = FloatVector.fromArray(sp, in, base + 8 * l).add(a8);
            a9 = FloatVector.fromArray(sp, in, base + 9 * l).add(a9);
            a10 = FloatVector.fromArray(sp, in, base + 10 * l).add(a10);
            a11 = FloatVector.fromArray(sp, in, base + 11 * l).add(a11);
            a12 = FloatVector.fromArray(sp, in, base + 12 * l).add(a12);
            a13 = FloatVector.fromArray(sp, in, base + 13 * l).add(a13);
            a14 = FloatVector.fromArray(sp, in, base + 14 * l).add(a14);
            a15 = FloatVector.fromArray(sp, in, base + 15 * l).add(a15);
            a16 = FloatVector.fromArray(sp, in, base + 16 * l).add(a16);
            a17 = FloatVector.fromArray(sp, in, base + 17 * l).add(a17);
            a18 = FloatVector.fromArray(sp, in, base + 18 * l).add(a18);
            a19 = FloatVector.fromArray(sp, in, base + 19 * l).add(a19);
        }
        a0.intoArray(out, 0 * l);
        a1.intoArray(out, 1 * l);
        a2.intoArray(out, 2 * l);
        a3.intoArray(out, 3 * l);
        a4.intoArray(out, 4 * l);
        a5.intoArray(out, 5 * l);
        a6.intoArray(out, 6 * l);
        a7.intoArray(out, 7 * l);
        a8.intoArray(out, 8 * l);
        a9.intoArray(out, 9 * l);
        a10.intoArray(out, 10 * l);
        a11.intoArray(out, 11 * l);
        a12.intoArray(out, 12 * l);
        a13.intoArray(out, 13 * l);
        a14.intoArray(out, 14 * l);
        a15.intoArray(out, 15 * l);
        a16.intoArray(out, 16 * l);
        a17.intoArray(out, 17 * l);
        a18.intoArray(out, 18 * l);
        a19.intoArray(out, 19 * l);
    }

    @Override
    protected LIRSuites createLIRSuites(OptionValues opts) {
        LIRSuites suites = super.createLIRSuites(opts);
        suites.getFinalCodeAnalysisStage().appendPhase(new LIRPhase<FinalCodeAnalysisContext>() {
            @Override
            protected CharSequence createName() {
                return "HighRegisterSpillVerification";
            }

            @Override
            protected void run(TargetDescription target, LIRGenerationResult lirGenRes, FinalCodeAnalysisContext context) {
                boolean simd = false;
                boolean high = false;
                int spills = 0;
                var lir = lirGenRes.getLIR();
                for (int blockId : lir.codeEmittingOrder()) {
                    if (blockId == INVALID_BLOCK_ID) {
                        continue;
                    }
                    BasicBlock<?> block = lir.getBlockById(blockId);
                    for (LIRInstruction instr : lir.getLIRforBlock(block)) {
                        boolean[] flags = collect(instr);
                        simd |= flags[0];
                        high |= flags[1];
                        if (ValueMoveOp.isValueMoveOp(instr)) {
                            ValueMoveOp move = ValueMoveOp.asValueMoveOp(instr);
                            if (isSimdStackValue(move.getResult()) || isSimdStackValue(move.getInput())) {
                                spills++;
                            }
                        }
                    }
                }
                if (simd) {
                    // Record the kernel compilation (ignore any non-vector compiles).
                    sawSimd = true;
                    usesHighRegister = high;
                    simdSpillMoves = spills;
                }
            }
        });
        return suites;
    }

    @Test
    public void noSpillingWithHighRegisters() {
        assumeTrue("test is AMD64 specific", getTarget().arch instanceof AMD64);
        assumeTrue("requires full AVX-512 (otherwise xmm16-31 are not allocatable)",
                        AMD64BaseAssembler.supportsFullAVX512(((AMD64) getTarget().arch).getFeatures()));
        VectorArchitecture vectorArch = ((VectorLoweringProvider) getProviders().getLowerer()).getVectorArchitecture();
        int len = FloatVector.SPECIES_PREFERRED.length();
        assumeTrue("requires vectorized float add at the preferred length",
                        vectorArch.getSupportedVectorArithmeticLength(StampFactory.forKind(JavaKind.Float), len, IntegerStamp.OPS.getAdd()) == len);

        int iters = 4;
        float[] in = new float[iters * ACCUMULATORS * len];
        for (int i = 0; i < in.length; i++) {
            in[i] = i * 0.5f;
        }
        ArgSupplier out = () -> new float[ACCUMULATORS * len];
        test("manyAccumulators", in, out, iters);

        assertTrue("kernel did not vectorize - cannot evaluate register usage", sawSimd);
        assertTrue("the kernel must use at least one AVX-512 high register (xmm16-31); "
                        + "register allocation is not using the extended register file", usesHighRegister);
        assertDeepEquals("number of SIMD spill moves in a " + ACCUMULATORS + "-accumulator kernel", 0, simdSpillMoves);
    }

    /** @return {@code [usesSimdRegister, usesHighRegister]} for the instruction's outputs. */
    private static boolean[] collect(LIRInstruction instr) {
        boolean[] flags = {false, false};
        instr.forEachOutput((value, _, _) -> {
            if (ValueUtil.isRegister(value)) {
                var reg = ValueUtil.asRegister(value);
                if (AMD64.XMM.equals(reg.getRegisterCategory())) {
                    flags[0] = true;
                }
                if (AMD64BaseAssembler.isAVX512Register(reg)) {
                    flags[1] = true;
                }
            }
            return value;
        });
        return flags;
    }

    private static boolean isSimdStackValue(Value value) {
        if (!LIRValueUtil.isStackSlotValue(value)) {
            return false;
        }
        PlatformKind kind = value.getValueKind().getPlatformKind();
        return kind instanceof AMD64Kind && ((AMD64Kind) kind).isXMM() && ((AMD64Kind) kind).getVectorLength() > 1;
    }
}
