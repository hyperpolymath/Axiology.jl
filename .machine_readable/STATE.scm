;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Axiology.jl
;; Media-Type: application/vnd.state+scm

(define-state Axiology.jl
  (metadata
    (version "0.1.0")
    (schema-version "1.0.0")
    (created "2026-02-07")
    (updated "2026-02-07")
    (project "Axiology.jl")
    (repo "hyperpolymath/Axiology.jl"))

  (project-context
    (name "Axiology.jl")
    (tagline "Value theory for machine learning")
    (tech-stack (julia ml-ethics optimization verification)))

  (current-position
    (phase "minimal-scaffold")
    (overall-completion 10)
    (components
      (value-types "Basic abstract Value type and concrete types (Fairness, Welfare, Profit, Efficiency)"))
    (working-features
      "Minimal placeholder implementations of satisfy, maximize, verify_value"))

  (route-to-mvp
    (milestones
      ((name "Core Value System")
       (status "in-progress")
       (completion 10)
       (items
         ("Define Value type hierarchy" . done)
         ("Implement fairness metrics" . todo)
         ("Implement welfare functions" . todo)
         ("Implement profit optimization" . todo)
         ("Implement efficiency measures" . todo)))
      ((name "ML Integration")
       (status "todo")
       (completion 0)
       (items
         ("Value-constrained optimization" . todo)
         ("Model verification against values" . todo)
         ("Trade-off analysis" . todo)))
      ((name "Documentation & Examples")
       (status "todo")
       (completion 0)
       (items
         ("Usage examples" . todo)
         ("Case studies" . todo)
         ("Integration guides" . todo)))))

  (blockers-and-issues
    (critical ())
    (high
      ("Need to define precise fairness metrics" . "What fairness definitions to implement?")
      ("Need to clarify welfare functions" . "Utilitarian? Rawlsian? Other?"))
    (medium ())
    (low ()))

  (critical-next-actions
    (immediate
      "Define fairness metrics (demographic parity, equalized odds, etc.)"
      "Implement basic welfare functions (utilitarian sum, Rawlsian max-min)")
    (this-week
      "Add profit optimization with value constraints"
      "Implement efficiency measures (Pareto, Kaldor-Hicks)")
    (this-month
      "ML model verification infrastructure"
      "Integration with Flux.jl and other ML frameworks"))

  (session-history ()))

;; Helper functions
(define (get-completion-percentage state)
  (current-position 'overall-completion state))

(define (get-blockers state severity)
  (blockers-and-issues severity state))

(define (get-milestone state name)
  (find (lambda (m) (equal? (car m) name))
        (route-to-mvp 'milestones state)))
