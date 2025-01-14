module AdditionNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "AdditionNode" begin
    @testset "Creation" begin
        node = make_node(+)

        @test functionalform(node) === +
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in1, :in2)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_in1_in2,)
        @test metadata(node) === nothing

        node = make_node(+, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === +
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in1, :in2)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_in1_in2,)
        @test metadata(node) === 1

        node = make_node(+, FactorNodeCreationOptions(MeanField(), nothing, nothing))

        @test functionalform(node) === +
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in1, :in2)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_in1_in2,)
        @test metadata(node) === nothing
    end
end
end
