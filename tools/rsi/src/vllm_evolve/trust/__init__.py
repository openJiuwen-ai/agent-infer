"""Trust — code-safety checks for evolved policy code.

The live entrypoint is ``trust.safety`` (AST safety / signature checks), used by the verify path.
The legacy multi-layer ``TrustChain`` orchestrator (chain.py / constraint.py) belonged to the
removed population-evolution engine and has been deleted.
"""
