# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

# Regression tests for undefined scores, the :protected_attributes alias,
# and public-API documentation coverage.

@testset "Hardening" begin
    @testset "Zero-target scores throw instead of returning Inf/NaN" begin
        @test_throws ArgumentError value_score(Profit(), Dict(:profit => 100.0))
        @test_throws ArgumentError value_score(Profit(target = 0.0), Dict(:profit => 0.0))
        @test_throws ArgumentError value_score(Efficiency(metric = :kaldor_hicks, target = 0.0),
                                               Dict(:net_gain => 1.0))
        @test_throws ArgumentError value_score(Efficiency(metric = :computation_time, target = 0.0),
                                               Dict(:computation_time => 1.0))
        @test_throws ArgumentError weighted_score(Value[Profit()], Dict(:profit => 100.0))
        # satisfy is unaffected: a zero profit target is a meaningful floor.
        @test satisfy(Profit(), Dict(:profit => 100.0))
    end

    @testset "Zero fairness threshold demands exact parity" begin
        f = Fairness(metric = :demographic_parity, threshold = 0.0)
        parity = Dict(:predictions => [1, 0, 1, 0], :protected => [:a, :a, :b, :b])
        skewed = Dict(:predictions => [1, 1, 1, 0], :protected => [:a, :a, :b, :b])
        @test value_score(f, parity) == 1.0
        @test value_score(f, skewed) == 0.0
        @test !isnan(weighted_score(Value[f], parity))
    end

    @testset "value_score accepts :protected_attributes like satisfy does" begin
        f = Fairness(metric = :demographic_parity, threshold = 0.5)
        a = Dict(:predictions => [1, 1, 1, 0], :protected => [:a, :a, :b, :b])
        b = Dict(:predictions => [1, 1, 1, 0], :protected_attributes => [:a, :a, :b, :b])
        @test value_score(f, a) == value_score(f, b) == 0.0
        @test satisfy(f, a) == satisfy(f, b)
    end

    @testset "Every exported name is documented" begin
        # Base.Docs.hasdoc only exists from Julia 1.11; the docs registry
        # (Base.Docs.meta) answers the same question on every supported version.
        docs = Base.Docs.meta(Axiology)
        undocumented = [n for n in names(Axiology)
                        if n !== :Axiology && !haskey(docs, Base.Docs.Binding(Axiology, n))]
        @test isempty(undocumented)
        # Positive control: the check must be able to see a missing docstring.
        @test !haskey(docs, Base.Docs.Binding(Axiology, :_no_such_binding))
    end

    @testset "ECHIDNA prove-result receipts" begin
        s = Safety(invariant = "no harmful recommendations", critical = true)
        # The documented example object from echidna docs/PROVE-RESULT-CONTRACT.adoc.
        ok() = Dict{String,Any}("duration_ms" => 1, "echidna_version" => "x",
                                "goal" => "ok.twasm", "message" => "PROVED in 0ms",
                                "prover" => "TypedWasm", "schema" => PROVE_RESULT_SCHEMA,
                                "status" => "verified",
                                "trust" => Dict{String,Any}("axioms" => Any[], "confidence" => nothing))
        @test verify_receipt(s, ok())
        @test verify_value(s, ok())                       # dispatches to the receipt check
        @test verify_receipt(s, ok(); goal = "ok.twasm")
        @test !verify_receipt(s, ok(); goal = "other.lean")
        for st in ("failed", "error", "timeout", "unknown")
            r = ok(); r["status"] = st
            @test !verify_receipt(s, r)
        end
        r = ok(); r["prover"] = ""
        @test !verify_receipt(s, r)
        r = ok(); r["trust"]["axioms"] = Any["sorry"]
        @test !verify_receipt(s, r)
        @test verify_receipt(s, r; allow_axioms = ["sorry"])
        # Symbol-keyed parses (e.g. JSON3) are read the same way.
        sym = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in ok())
        sym[:trust] = Dict{Symbol,Any}(:axioms => Any[], :confidence => nothing)
        @test verify_receipt(s, sym)
        # Malformed receipts are rejected, not read as failures.
        r = ok(); r["schema"] = "echidna.prove.result/2"
        @test_throws ArgumentError verify_receipt(s, r)
        r = ok(); r["status"] = "proved"
        @test_throws ArgumentError verify_receipt(s, r)
        r = ok(); delete!(r, "goal")
        @test_throws ArgumentError verify_receipt(s, r)
        r = ok(); r["trust"] = Dict{String,Any}("axioms" => "sorry")
        @test_throws ArgumentError verify_receipt(s, r)
        r = ok(); r["unexpected_future_field"] = 42   # unknown fields are ignored
        @test verify_receipt(s, r)
        # Plain attestations keep their old meaning.
        @test verify_value(Welfare(), Dict(:verified => true))
    end

    @testset "safety_verdict keeps missing evidence visible" begin
        s = Safety(invariant = "test")
        @test safety_verdict(s, Dict()) === :unresolved
        @test safety_verdict(s, Dict(:is_safe => true)) === :unresolved
        @test safety_verdict(s, Dict(:is_safe => true, :invariant_holds => true)) === :entailed
        @test safety_verdict(s, Dict(:is_safe => false)) === :refuted
        @test safety_verdict(s, Dict(:is_safe => true, :invariant_holds => false)) === :refuted
        @test_throws ArgumentError safety_verdict(s, Dict(:is_safe => 1))
        @test satisfy(s, Dict())                          # unchanged: absent keys count as safe
    end

    @testset "weighted_score is lossy (distinct profiles, same score)" begin
        values = Value[Efficiency(metric = :pareto, weight = 1.0),
                       Safety(invariant = "test", weight = 1.0)]
        a = Dict(:is_pareto_efficient => true, :is_safe => false)
        b = Dict(:is_pareto_efficient => false, :is_safe => true)
        @test [value_score(v, a) for v in values] != [value_score(v, b) for v in values]
        @test weighted_score(values, a) == weighted_score(values, b) == 0.5
        # ...and the Pareto comparison still tells them apart: neither dominates.
        @test !dominated(a, b, values) && !dominated(b, a, values)
    end

    @testset "Module docstring example holds" begin
        fairness = Fairness(metric = :demographic_parity, protected_attributes = [:group],
                            threshold = 0.05)
        state = Dict(:predictions => [1, 0, 1, 1, 0, 1], :protected => [:a, :a, :b, :b, :b, :a])
        @test satisfy(fairness, state)
        values = [Welfare(metric = :utilitarian, weight = 0.5),
                  Fairness(metric = :demographic_parity, threshold = 0.1, weight = 0.5)]
        candidates = [
            Dict(:utilities => [3.0, 3.0], :predictions => [1, 1], :protected => [:a, :b]),
            Dict(:utilities => [9.0, 0.0], :predictions => [1, 0], :protected => [:a, :b]),
        ]
        @test length(pareto_frontier(candidates, values)) == 2
    end
end
