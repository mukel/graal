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

import static jdk.graal.compiler.asm.amd64.AMD64BaseAssembler.isAVX512Register;
import static jdk.graal.compiler.asm.amd64.AMD64MacroAssembler.ExtendMode.ZERO_EXTEND;
import static jdk.vm.ci.code.ValueUtil.asRegister;
import static jdk.vm.ci.code.ValueUtil.isIllegal;
import static jdk.vm.ci.code.ValueUtil.isRegister;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import org.junit.Test;

import jdk.graal.compiler.asm.amd64.AMD64BaseAssembler;
import jdk.graal.compiler.asm.amd64.AVXKind.AVXSize;
import jdk.graal.compiler.core.common.LIRKind;
import jdk.graal.compiler.core.common.Stride;
import jdk.graal.compiler.lir.LIRInstruction;
import jdk.graal.compiler.lir.Variable;
import jdk.graal.compiler.lir.amd64.AMD64ArrayIndexOfOp;
import jdk.graal.compiler.lir.amd64.AMD64ArrayRegionCompareToOp;
import jdk.graal.compiler.lir.amd64.AMD64CalcStringAttributesOp;
import jdk.graal.compiler.lir.amd64.AMD64CodepointIndexToByteIndexOp;
import jdk.graal.compiler.lir.amd64.AMD64IndexOfZeroOp;
import jdk.graal.compiler.lir.amd64.AMD64VectorizedHashCodeOp;
import jdk.graal.compiler.lir.amd64.AMD64VectorizedMismatchOp;
import jdk.graal.compiler.lir.gen.LIRGeneratorTool;
import jdk.graal.compiler.replacements.nodes.StringCodepointIndexToByteIndexNode.InputEncoding;
import jdk.vm.ci.amd64.AMD64;
import jdk.vm.ci.amd64.AMD64.CPUFeature;
import jdk.vm.ci.amd64.AMD64Kind;
import jdk.vm.ci.code.Register;
import jdk.vm.ci.code.TargetDescription;
import jdk.vm.ci.meta.AllocatableValue;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.Value;
import jdk.vm.ci.meta.ValueKind;

/**
 * Regression coverage for complex AMD64 vector LIR ops that still emit VEX-only instructions.
 * Under full AVX-512 these ops must expose fixed xmm0-15 temps to register allocation; otherwise
 * VEX emission can fail if an allocator-chosen temp lands in xmm16-31.
 */
public class AMD64ComplexVectorLowTempTest {

    private static final EnumSet<CPUFeature> AVX2_FEATURES = EnumSet.of(
                    CPUFeature.SSE, CPUFeature.SSE2, CPUFeature.SSSE3, CPUFeature.SSE4_1, CPUFeature.SSE4_2,
                    CPUFeature.POPCNT, CPUFeature.BMI1, CPUFeature.AVX, CPUFeature.AVX2);

    /** AVX2 base plus the production "full AVX-512" set the pinning keys off, kept in sync via reuse. */
    private static final EnumSet<CPUFeature> FULL_AVX512_FEATURES = fullAVX512Features();

    private static EnumSet<CPUFeature> fullAVX512Features() {
        EnumSet<CPUFeature> features = EnumSet.copyOf(AVX2_FEATURES);
        features.addAll(AMD64BaseAssembler.FULL_AVX512_FEATURES);
        return features;
    }

    @Test
    public void fullAVX512PinsVexOnlyTempsToLowXmmRegisters() {
        for (LIRInstruction op : createOps(tool(avx512Target()), FULL_AVX512_FEATURES)) {
            List<Value> xmmTemps = xmmTemps(op);
            assertFalse(op.name() + " should expose fixed low vector temps under full AVX-512", xmmTemps.isEmpty());
            for (Value temp : xmmTemps) {
                Register register = asRegister(temp);
                assertFalse(op.name() + " temp must be encodable by VEX: " + register, isAVX512Register(register));
            }
        }
    }

    @Test
    public void avx2KeepsVexOnlyTempsAllocatable() {
        for (LIRInstruction op : createOps(tool(avx2Target()), AVX2_FEATURES)) {
            assertTrue(op.name() + " should not pin vector temps when high registers are unavailable", xmmTemps(op).isEmpty());
            assertFalse(op.name() + " should still have allocator-controlled vector temps", vectorVariables(op).isEmpty());
        }
    }

    private static List<LIRInstruction> createOps(LIRGeneratorTool tool, EnumSet<CPUFeature> features) {
        List<LIRInstruction> ops = new ArrayList<>();
        Value q0 = qword(AMD64.rax);
        Value q1 = qword(AMD64.rbx);
        Value q2 = qword(AMD64.rdx);
        Value q3 = qword(AMD64.r8);
        Value d0 = dword(AMD64.r10);
        Value d1 = dword(AMD64.r11);
        Value d2 = dword(AMD64.r12);
        Value d3 = dword(AMD64.r13);
        Value d4 = dword(AMD64.r14);

        for (Stride stride : new Stride[]{Stride.S1, Stride.S2, Stride.S4}) {
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.MatchAny, -1, 4, tool, features, d0, q0, q1, d1, d2, d1, d2, d3, d4));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.MatchRange, -1, 4, tool, features, d0, q0, q1, d1, d2, d1, d2, d3, d4));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.MatchRangeForeignEndian, -1, 4, tool, features, d0, q0, q1, d1, d2, d1, d2, d3, d4));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.WithMask, -1, 2, tool, features, d0, q0, q1, d1, d2, d1, d2, Value.ILLEGAL, Value.ILLEGAL));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.FindTwoConsecutive, -1, 2, tool, features, d0, q0, q1, d1, d2, d1, d2, Value.ILLEGAL, Value.ILLEGAL));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.FindTwoConsecutiveWithMask, -1, 4, tool, features, d0, q0, q1, d1, d2, d1, d2, d3, d4));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.Table, -1, 1, tool, features, d0, q0, q1, d1, d2, q3, Value.ILLEGAL, Value.ILLEGAL, Value.ILLEGAL));
            ops.add(new AMD64ArrayIndexOfOp(stride, LIRGeneratorTool.ArrayIndexOfVariant.TableForeignEndian, -1, 1, tool, features, d0, q0, q1, d1, d2, q3, Value.ILLEGAL, Value.ILLEGAL, Value.ILLEGAL));

            ops.add(AMD64IndexOfZeroOp.movParamsAndCreate(stride, tool, features, q1, q0));
        }

        ops.add(new AMD64ArrayRegionCompareToOp(tool, Stride.S4, Stride.S1, features, d0, q0, q1, q2, q3, d1, Value.ILLEGAL, ZERO_EXTEND));
        ops.add(new AMD64ArrayRegionCompareToOp(tool, null, null, features, d0, q0, q1, q2, q3, d1, d2, ZERO_EXTEND));

        for (LIRGeneratorTool.CalcStringAttributesEncoding encoding : LIRGeneratorTool.CalcStringAttributesEncoding.values()) {
            ops.add(new AMD64CalcStringAttributesOp(tool, encoding, features, q0, q1, d1, q2, false));
            ops.add(new AMD64CalcStringAttributesOp(tool, encoding, features, q0, q1, d1, q2, true));
        }

        for (InputEncoding encoding : InputEncoding.values()) {
            ops.add(new AMD64CodepointIndexToByteIndexOp(tool, encoding, features, q0, q1, d1, d2, d0));
        }

        ops.add(new AMD64VectorizedHashCodeOp(tool, features, (AllocatableValue) d0, (AllocatableValue) q0, (AllocatableValue) d1, (AllocatableValue) d2, JavaKind.Int));
        ops.add(new AMD64VectorizedMismatchOp(tool, features, q0, q1, q2, q3, AMD64.rcx.asValue(LIRKind.value(AMD64Kind.DWORD))));
        return ops;
    }

    private static List<Value> xmmTemps(LIRInstruction op) {
        List<Value> temps = new ArrayList<>();
        op.visitEachTemp((value, mode, flags) -> {
            if (!isIllegal(value) && isRegister(value) && asRegister(value).getRegisterCategory().equals(AMD64.XMM)) {
                temps.add(value);
            }
        });
        return temps;
    }

    private static List<Value> vectorVariables(LIRInstruction op) {
        List<Value> temps = new ArrayList<>();
        op.visitEachTemp((value, mode, flags) -> {
            if (value instanceof Variable && value.getPlatformKind() instanceof AMD64Kind kind && kind.isXMM()) {
                temps.add(value);
            }
        });
        return temps;
    }

    /**
     * A minimal {@link LIRGeneratorTool} stub that supports just enough for constructing the ops
     * under test (which only call these methods from their constructors): {@code target},
     * {@code getMaxVectorSize}, {@code newVariable}, {@code toRegisterKind}, {@code asAllocatable},
     * and {@code emitMove}. {@code newVariable} returns plain {@link Variable}s so that any temp the
     * op does NOT pin shows up as an allocator-controlled variable. Any other method returns a type
     * default; if a future op constructor starts calling an unhandled tool method, extend this
     * allow-list (a {@code null}/{@code 0} default may otherwise surface as an NPE here).
     */
    private static LIRGeneratorTool tool(TargetDescription target) {
        InvocationHandler handler = new InvocationHandler() {
            private int variableIndex;

            @Override
            public Object invoke(Object proxy, Method method, Object[] args) {
                return switch (method.getName()) {
                    case "target" -> target;
                    case "getMaxVectorSize" -> AVXSize.YMM;
                    case "newVariable" -> new Variable((ValueKind<?>) args[0], variableIndex++);
                    case "toRegisterKind" -> args[0];
                    case "asAllocatable" -> (AllocatableValue) args[0];
                    case "emitMove" -> method.getReturnType() == Void.TYPE ? null : new Variable(((Value) args[0]).getValueKind(), variableIndex++);
                    case "toString" -> "TestLIRGeneratorTool";
                    default -> defaultValue(method.getReturnType());
                };
            }
        };
        return (LIRGeneratorTool) Proxy.newProxyInstance(LIRGeneratorTool.class.getClassLoader(), new Class<?>[]{LIRGeneratorTool.class}, handler);
    }

    private static Object defaultValue(Class<?> type) {
        if (type == Void.TYPE) {
            return null;
        } else if (type == Boolean.TYPE) {
            return false;
        } else if (type == Byte.TYPE) {
            return (byte) 0;
        } else if (type == Short.TYPE) {
            return (short) 0;
        } else if (type == Character.TYPE) {
            return (char) 0;
        } else if (type == Integer.TYPE) {
            return 0;
        } else if (type == Long.TYPE) {
            return 0L;
        } else if (type == Float.TYPE) {
            return 0f;
        } else if (type == Double.TYPE) {
            return 0d;
        }
        return null;
    }

    private static TargetDescription avx512Target() {
        return target(FULL_AVX512_FEATURES);
    }

    private static TargetDescription avx2Target() {
        return target(AVX2_FEATURES);
    }

    private static TargetDescription target(EnumSet<CPUFeature> features) {
        return new TargetDescription(new AMD64(features), true, 16, 4096, true);
    }

    private static Value qword(Register register) {
        return register.asValue(LIRKind.value(AMD64Kind.QWORD));
    }

    private static Value dword(Register register) {
        return register.asValue(LIRKind.value(AMD64Kind.DWORD));
    }
}
