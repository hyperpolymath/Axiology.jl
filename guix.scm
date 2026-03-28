; SPDX-License-Identifier: PMPL-1.0-or-later
;; guix.scm — GNU Guix package definition for Axiology.jl
;; Usage: guix shell -f guix.scm

(use-modules (guix packages)
             (guix build-system gnu)
             (guix licenses))

(package
  (name "Axiology.jl")
  (version "0.1.0")
  (source #f)
  (build-system gnu-build-system)
  (synopsis "Axiology.jl")
  (description "Axiology.jl — part of the hyperpolymath ecosystem.")
  (home-page "https://github.com/hyperpolymath/Axiology.jl")
  (license ((@@ (guix licenses) license) "PMPL-1.0-or-later"
             "https://github.com/hyperpolymath/palimpsest-license")))
