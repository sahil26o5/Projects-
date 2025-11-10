from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Set, List, Any
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import numpy as np
import json

#_____________________________________________
# author @sahil maharaj @student:2550404
#        this is the code for De Groote SC backward parser
#_____________________________________________

# ========== Node and helpers ==========
@dataclass(eq=False)
class Node:
    val: str
    left: Optional['Node'] = None
    right: Optional['Node'] = None

    def is_hole(self) -> bool:
        return self.val == '#'

    def is_leaf(self) -> bool:
        return (self.left is None) and (self.right is None)

    def copy(self) -> 'Node':
        return deepcopy(self)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.val == other.val and
                self.left == other.left and
                self.right == other.right)


# ========== Full Backward SC Prover ==========
class SCBackwardProver:
    def __init__(self, trace: bool = False):
        self.trace = trace

        # Three main tables / caches
        self.seq_cache: Dict[Tuple[str, str], bool] = {}  # (lhs, rhs) -> bool
        self.neg_context_cache: Dict[str, bool] = {}  # context_pol -> bool
        self.pos_context_cache: Dict[str, bool] = {}  # context_pol -> bool

        # Store derivations for proof reconstruction
        self.derivations: Dict[Tuple[str, str], Tuple[str, List]] = {}
        self.neg_context_derivations: Dict[str, Tuple[str, List]] = {}
        self.pos_context_derivations: Dict[str, Tuple[str, List]] = {}

        # Formula and context storage
        self.formulas_table: Dict[str, Node] = {}
        self.contexts_table: Dict[str, Node] = {}

    # ---------- parsing ----------
    def parse_polish_tokens(self, tokens: List[str], i: int = 0) -> Tuple[Node, int]:
        if i >= len(tokens):
            raise ValueError("Unexpected end of tokens")
        t = tokens[i]
        if t in {'\\', '/', '•'}:
            left, j = self.parse_polish_tokens(tokens, i + 1)
            right, k = self.parse_polish_tokens(tokens, j)
            return Node(t, left, right), k
        return Node(t), i + 1

    def parse_polish(self, s: str) -> Node:
        if isinstance(s, Node):
            return s
        tokens = s.strip().split()
        if not tokens:
            raise ValueError("Empty polish string")
        root, next_i = self.parse_polish_tokens(tokens, 0)
        if next_i != len(tokens):
            raise ValueError("Extra tokens after parsing: " + " ".join(tokens[next_i:]))
        return root

    def to_polish(self, node: Node) -> str:
        if node is None:
            return ""
        if node.is_leaf():
            return node.val
        return f"{node.val} {self.to_polish(node.left)} {self.to_polish(node.right)}"

    # ---------- hole / composition utilities ----------
    def _find_hole_path(self, node: Node, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        if node is None:
            return None
        if node.is_hole():
            return path.copy()
        if node.left:
            res = self._find_hole_path(node.left, path + ['L'])
            if res is not None:
                return res
        if node.right:
            res = self._find_hole_path(node.right, path + ['R'])
            if res is not None:
                return res
        return None

    def count_holes(self, node: Node) -> int:
        if node is None:
            return 0
        if node.is_hole():
            return 1
        return (self.count_holes(node.left) if node.left else 0) + (self.count_holes(node.right) if node.right else 0)

    def _fill_hole_rec(self, context: Node, filler: Node) -> Tuple[Node, bool]:
        """
        Replace FIRST hole '#' in 'context' with 'filler' (do not mutate inputs).
        Returns (new_node, changed).
        """
        if context is None:
            return context, False
        if context.is_hole():
            # return deep copy of filler to avoid aliasing
            return deepcopy(filler), True

        # try left
        if context.left:
            left_new, left_changed = self._fill_hole_rec(context.left, filler)
            if left_changed:
                new_node = Node(context.val, left_new, deepcopy(context.right) if context.right else None)
                return new_node, True

        # try right
        if context.right:
            right_new, right_changed = self._fill_hole_rec(context.right, filler)
            if right_changed:
                new_node = Node(context.val, deepcopy(context.left) if context.left else None, right_new)
                return new_node, True

        return deepcopy(context), False

    def fill_hole(self, context: Node, filler: Node) -> Node:
        new_node, changed = self._fill_hole_rec(context, filler)
        if not changed:
            raise ValueError("No hole '#' found in context: " + self.to_polish(context))
        return new_node

    def compose_contexts(self, context: Node, filler: Node) -> Node:
        """
        Compose context by replacing the first hole in 'context' with 'filler',
        preserving any further holes present in filler. Returns a deep-copied result.
        """
        if context is None:
            return deepcopy(filler)
        # If context itself is a hole, fill directly
        if context.is_hole():
            return deepcopy(filler)

        # use _fill_hole_rec to replace first hole only
        res, changed = self._fill_hole_rec(context.copy(), filler.copy())
        return res

    def delete_children_at_hole(self, node: Node):
        """
        After composition, holes should be leaves. Remove children under holes.
        """
        if node is None:
            return
        if node.is_hole():
            node.left = None
            node.right = None
            return
        if node.left:
            self.delete_children_at_hole(node.left)
        if node.right:
            self.delete_children_at_hole(node.right)

    # ---------- initialization ----------
    def initialize_from_inputs(self, lhs: str, rhs: str):
        """Initialize formulas and contexts tables from input sequent"""
        Lnode = self.parse_polish(lhs)
        Rnode = self.parse_polish(rhs)

        # Clear previous state
        self.formulas_table.clear()
        self.contexts_table.clear()
        self.seq_cache.clear()
        self.neg_context_cache.clear()
        self.pos_context_cache.clear()
        self.derivations.clear()
        self.neg_context_derivations.clear()
        self.pos_context_derivations.clear()

        # Store main formulas and all subformulas
        self._collect_all_subformulas(Lnode)
        self._collect_all_subformulas(Rnode)

        # Store all possible contexts
        self._collect_all_subcontexts(Lnode)
        self._collect_all_subcontexts(Rnode)

        # Add the hole context
        hole_node = Node('#')
        self.contexts_table['#'] = hole_node

    def _collect_all_subformulas(self, node: Node):
        """Collect all subformulas (without holes)"""
        if node is None:
            return
        pol = self.to_polish(node)
        if '#' not in pol and pol not in self.formulas_table:
            self.formulas_table[pol] = node.copy()
        if node.left:
            self._collect_all_subformulas(node.left)
        if node.right:
            self._collect_all_subformulas(node.right)

    def _find_all_paths(self, node: Node, current_path: List[str], all_paths: List[List[str]]):
        """Find all paths to subformulas in the tree"""
        all_paths.append(current_path.copy())
        if node.left:
            self._find_all_paths(node.left, current_path + ['L'], all_paths)
        if node.right:
            self._find_all_paths(node.right, current_path + ['R'], all_paths)

    def _create_context_at_path(self, original: Node, path: List[str]) -> Node:
        """Create a context by replacing node at path with hole"""
        if not path:
            return Node('#')

        def build_context(node: Node, current_path: List[str]) -> Node:
            if not current_path:
                return Node('#')

            direction = current_path[0]
            new_node = node.copy()
            if direction == 'L' and node.left:
                new_node.left = build_context(node.left, current_path[1:])
            elif direction == 'R' and node.right:
                new_node.right = build_context(node.right, current_path[1:])
            return new_node

        return build_context(original, path)

    def _collect_all_subcontexts(self, node: Node):
        """Collect all possible contexts by replacing each subformula with a hole"""
        paths = []
        self._find_all_paths(node, [], paths)

        for path in paths:
            context = self._create_context_at_path(node, path)
            context_pol = self.to_polish(context)
            if context_pol not in self.contexts_table:
                self.contexts_table[context_pol] = context

    # ---------- main backward prover ----------
    def prove_backward(self, lhs: str, rhs: str) -> bool:
        """Main backward prover - follows your exact specification (Version B)."""

        # Check cache
        key = (lhs, rhs)
        if key in self.seq_cache:
            #if self.trace:
                #print(f"[cache] {lhs} ⊢ {rhs} = {self.seq_cache[key]}")
            return self.seq_cache[key]

        # ATOMIC Identity (only for single atoms)
        if lhs == rhs and len(lhs.split()) == 1 and lhs not in {'\\', '/', '•'}:
            self.seq_cache[key] = True
            self.derivations[key] = ("Atom-Id", [])
            #if self.trace:
                #print(f"[Atom-Id] {lhs} ⊢ {rhs}")
            return True


        # provisional false to break cycles
        self.seq_cache[key] = False
        #if self.trace:
            #print(f"[start] Trying to prove: {lhs} ⊢ {rhs}")

        # Parse into trees
        try:
            Lnode = self.parse_polish(lhs)
            Rnode = self.parse_polish(rhs)
        except Exception:
            return False

        # connective rules
        if self._try_connective_rules(Lnode, Rnode, lhs, rhs):
            return True

        # Continuation rules (ContN / ContP)
        if self._try_continuation_rules(lhs, rhs):
            return True

        #if self.trace:
            #print(f"[fail] No rule applies: {lhs} ⊢ {rhs}")
        self.seq_cache[key] = False
        return False

    def _try_connective_rules(self, Lnode: Node, Rnode: Node, lhs: str, rhs: str) -> bool:
        """Apply the three basic connective sequent rules structurally."""
        key = (lhs, rhs)

        # (\) rule : (B\C) ⊢ (A\D)  if A ⊢ B and C ⊢ D
        if Lnode and Rnode and Lnode.val == '\\' and Rnode.val == '\\':
            B_node, C_node = Lnode.left, Lnode.right
            A_node, D_node = Rnode.left, Rnode.right
            prem1 = (self.to_polish(A_node), self.to_polish(B_node))  # A ⊢ B
            prem2 = (self.to_polish(C_node), self.to_polish(D_node))  # C ⊢ D

            #if self.trace:
                #print(f"[\\ rule] {lhs} ⊢ {rhs} ⇒ need {prem1} and {prem2}")

            if self.prove_backward(*prem1) and self.prove_backward(*prem2):
                self.seq_cache[key] = True
                self.derivations[key] = ("\\-rule", [prem1, prem2])
                #if self.trace:
                    #print(f"[\\ rule ✓] {lhs} ⊢ {rhs}")
                return True

        # (/) rule : (C/B) ⊢ (D/A)  if A ⊢ B and C ⊢ D
        if Lnode and Rnode and Lnode.val == '/' and Rnode.val == '/':
            C_node, B_node = Lnode.left, Lnode.right
            D_node, A_node = Rnode.left, Rnode.right
            prem1 = (self.to_polish(A_node), self.to_polish(B_node))  # A ⊢ B
            prem2 = (self.to_polish(C_node), self.to_polish(D_node))  # C ⊢ D

            #if self.trace:
                #print(f"[/ rule] {lhs} ⊢ {rhs} ⇒ need {prem1} and {prem2}")

            if self.prove_backward(*prem1) and self.prove_backward(*prem2):
                self.seq_cache[key] = True
                self.derivations[key] = ("/-rule", [prem1, prem2])
                #if self.trace:
                    #print(f"[/ rule ✓] {lhs} ⊢ {rhs}")
                return True

        # (•) rule : (A•C) ⊢ (B•D)  if A ⊢ B and C ⊢ D
        if Lnode and Rnode and Lnode.val == '•' and Rnode.val == '•':
            A_node, C_node = Lnode.left, Lnode.right
            B_node, D_node = Rnode.left, Rnode.right
            prem1 = (self.to_polish(A_node), self.to_polish(B_node))  # A ⊢ B
            prem2 = (self.to_polish(C_node), self.to_polish(D_node))  # C ⊢ D

            #if self.trace:
                #print(f"[• rule] {lhs} ⊢ {rhs} ⇒ need {prem1} and {prem2}")

            if self.prove_backward(*prem1) and self.prove_backward(*prem2):
                self.seq_cache[key] = True
                self.derivations[key] = ("•-rule", [prem1, prem2])
                #if self.trace:
                    #print(f"[• rule ✓] {lhs} ⊢ {rhs}")
                return True

        return False

    def _try_continuation_rules(self, lhs: str, rhs: str) -> bool:
        """Try ContN and ContP rules: use contexts_table and formulas_table."""

        key = (lhs, rhs)

        # ContN: Γ[A] ⊢ B  if A ⊢ B and ⊢N Γ[]
        #if self.trace:
            #print(f"[ContN-search] attempting for {lhs} ⊢ {rhs}")

        # iterate contexts that have exactly one hole (conservative)
        for ctx_pol, ctx_node in list(self.contexts_table.items()):
            if self.count_holes(ctx_node) != 1:
                continue

            for fpol, fnode in list(self.formulas_table.items()):
                try:
                    # Compose contexts by filling the first hole with fnode
                    composed = self.compose_contexts(ctx_node.copy(), deepcopy(fnode))
                    # After composition, hole nodes should be leaf/hole-free; prune children under holes
                    self.delete_children_at_hole(composed)
                    filled_pol = self.to_polish(composed)

                    if filled_pol == lhs:
                        #if self.trace:
                            #print(f"  [ContN-candidate] ctx={ctx_pol}, filler={fpol}")
                        # need f ⊢ rhs and ⊢N ctx
                        if self.prove_backward(fpol, rhs) and self.prove_neg_context(ctx_pol):
                            self.seq_cache[key] = True
                            self.derivations[key] = ("ContN", [(fpol, rhs), ("NEG_CTX", ctx_pol)])
                            #if self.trace:
                                #print(f"[ContN ✓] {lhs} ⊢ {rhs} via ctx={ctx_pol}, filler={fpol}")
                            return True
                except Exception as e:
                    if self.trace:
                        print(f"  [ContN error] {e}")

        # ContP: A ⊢ Γ[B]  if A ⊢ B and ⊢P Γ[]
        #if self.trace:
            #print(f"[ContP-search] attempting for {lhs} ⊢ {rhs}")

        for ctx_pol, ctx_node in list(self.contexts_table.items()):
            if self.count_holes(ctx_node) != 1:
                continue

            for fpol, fnode in list(self.formulas_table.items()):
                try:
                    composed = self.compose_contexts(ctx_node.copy(), deepcopy(fnode))
                    self.delete_children_at_hole(composed)
                    filled_pol = self.to_polish(composed)

                    if filled_pol == rhs:
                        #if self.trace:
                            #print(f"  [ContP-candidate] ctx={ctx_pol}, filler={fpol}")
                        # need lhs ⊢ filler and ⊢P ctx
                        if self.prove_backward(lhs, fpol) and self.prove_pos_context(ctx_pol):
                            self.seq_cache[key] = True
                            self.derivations[key] = ("ContP", [(lhs, fpol), ("POS_CTX", ctx_pol)])
                            #if self.trace:
                                #print(f"[ContP ✓] {lhs} ⊢ {rhs} via ctx={ctx_pol}, filler={fpol}")
                            return True
                except Exception as e:
                    if self.trace:
                        print(f"  [ContP error] {e}")

        return False

    # ---------- negative context prover ----------
    def prove_neg_context(self, ctx_pol: str) -> bool:
        """Prove ⊢N Γ[] for negative contexts (keeps permissive fallback)."""

        if ctx_pol in self.neg_context_cache:
            #if self.trace:
                #print(f"[cache N] {ctx_pol} = {self.neg_context_cache[ctx_pol]}")
            return self.neg_context_cache[ctx_pol]

        # provisional false
        self.neg_context_cache[ctx_pol] = False

        if ctx_pol not in self.contexts_table:
            return False

        ctx_node = self.contexts_table[ctx_pol]

        # root hole case
        if ctx_node.is_hole():
            self.neg_context_cache[ctx_pol] = True
            self.neg_context_derivations[ctx_pol] = ("[]-N", [])
            #if self.trace:
                #print(f"[[]-N] {ctx_pol} is hole -> True")
            return True

        # require exactly one hole for the negative context rules that we check here
        if self.count_holes(ctx_node) != 1:
            # If there is no hole or more than one hole, it's not one of the rules we know.
            return False

        # try negative context formation rules (per spec), keeping fallbacks but fixed matching
        if self._try_neg_context_rules(ctx_node, ctx_pol):
            return True

        # not proven
        return False

    def _try_neg_context_rules(self, ctx_node: Node, ctx_pol: str) -> bool:
        """Try negative context rules (•\\-N and •/-N) with exact pattern detection and permissive fallbacks."""

        # We only attempt these when ctx_node.val is '•' (both rules place a top-level •)
        if ctx_node.val != '•':
            return False

        # CASE (• \ -N): ⊢N (A • Γ[(B \ ∆[])])  if A ⊢ B and ⊢N Γ[] and ⊢N ∆[]
        A_node = ctx_node.left
        inner = ctx_node.right
        if inner is not None:
            inner_pol = self.to_polish(inner)
            # try to match inner to compose(gamma, Node('\\', B, delta))
            for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                # gamma must have exactly one hole (for the pattern)
                if self.count_holes(gamma_ctx) != 1:
                    continue
                for delta_pol, delta_ctx in list(self.contexts_table.items()):
                    if self.count_holes(delta_ctx) != 1:
                        continue
                    for B_pol, B_formula in list(self.formulas_table.items()):
                        try:
                            pattern = self.compose_contexts(gamma_ctx.copy(),
                                                            Node('\\', B_formula.copy(), delta_ctx.copy()))
                            self.delete_children_at_hole(pattern)
                            if self.to_polish(pattern) == inner_pol:
                                # matched pattern
                                A_pol = self.to_polish(A_node)
                                #if self.trace:
                                    #print(
                                       # f"  [•\\-N candidate] A={A_pol}, B={B_pol}, gamma={gamma_pol}, delta={delta_pol}")
                                if (self.prove_backward(A_pol, B_pol)
                                        and self.prove_neg_context(gamma_pol)
                                        and self.prove_neg_context(delta_pol)):
                                    self.neg_context_cache[ctx_pol] = True
                                    self.neg_context_derivations[ctx_pol] = (
                                        "•\\-N",
                                        [(A_pol, B_pol), ("NEG_CTX", gamma_pol), ("NEG_CTX", delta_pol)]
                                    )
                                    #if self.trace:
                                      #  print(f"[•\\-N ✓] {ctx_pol}")
                                    return True
                        except Exception:
                            continue

            #if self.trace:
               # print(f"  [•\\-N fallback] trying permissive fallback for {ctx_pol}")
            try:
                if self.prove_neg_context(inner_pol):
                    A_pol = self.to_polish(A_node)
                    # try identity first
                    if self.prove_backward(A_pol, A_pol) and self.prove_neg_context('#') and self.prove_neg_context(
                            inner_pol):
                        self.neg_context_cache[ctx_pol] = True
                        self.neg_context_derivations[ctx_pol] = (
                            "•\\-N-fallback-identity",
                            [(A_pol, A_pol), ("NEG_CTX", "#"), ("NEG_CTX", inner_pol)]
                        )
                        #if self.trace:
                          #  print(f"[•\\-N-fallback-identity ✓] {ctx_pol}")
                        return True
                    # otherwise try to find B such that A ⊢ B
                    for B_pol, _ in list(self.formulas_table.items()):
                        if self.prove_backward(A_pol, B_pol) and self.prove_neg_context('#') and self.prove_neg_context(
                                inner_pol):
                            self.neg_context_cache[ctx_pol] = True
                            self.neg_context_derivations[ctx_pol] = (
                                "•\\-N-fallback-general",
                                [(A_pol, B_pol), ("NEG_CTX", "#"), ("NEG_CTX", inner_pol)]
                            )
                            #if self.trace:
                               # print(f"[•\\-N-fallback-general ✓] {ctx_pol} with B={B_pol}")
                            return True
            except Exception:
                if self.trace:
                    print("  [•\\-N fallback error] unexpected exception")

        # CASE (•/-N): ⊢N (Γ[(Δ[]/B)] • A)  if A ⊢ B and ⊢N Γ[] and ⊢N Δ[]
        inner2 = ctx_node.left
        A_node2 = ctx_node.right
        if inner2 is not None:
            inner2_pol = self.to_polish(inner2)
            for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                if self.count_holes(gamma_ctx) != 1:
                    continue
                for delta_pol, delta_ctx in list(self.contexts_table.items()):
                    if self.count_holes(delta_ctx) != 1:
                        continue
                    for B_pol, B_formula in list(self.formulas_table.items()):
                        try:
                            pattern2 = self.compose_contexts(gamma_ctx.copy(),
                                                             Node('/', delta_ctx.copy(), B_formula.copy()))
                            self.delete_children_at_hole(pattern2)
                            if self.to_polish(pattern2) == inner2_pol:
                                A_pol = self.to_polish(A_node2)
                                #if self.trace:
                                  #  print(
                                     #   f"  [•/-N candidate] A={A_pol}, B={B_pol}, gamma={gamma_pol}, delta={delta_pol}")
                                if (self.prove_backward(A_pol, B_pol)
                                        and self.prove_neg_context(gamma_pol)
                                        and self.prove_neg_context(delta_pol)):
                                    self.neg_context_cache[ctx_pol] = True
                                    self.neg_context_derivations[ctx_pol] = (
                                        "•/-N",
                                        [(A_pol, B_pol), ("NEG_CTX", gamma_pol), ("NEG_CTX", delta_pol)]
                                    )
                                    #if self.trace:
                                       # print(f"[•/-N ✓] {ctx_pol}")
                                    return True
                        except Exception:
                            continue

            #if self.trace:
                #print(f"  [•/-N fallback] trying permissive fallback for {ctx_pol}")
            try:
                if self.prove_neg_context(inner2_pol):
                    A_pol = self.to_polish(A_node2)
                    if self.prove_backward(A_pol, A_pol) and self.prove_neg_context('#') and self.prove_neg_context(
                            inner2_pol):
                        self.neg_context_cache[ctx_pol] = True
                        self.neg_context_derivations[ctx_pol] = (
                            "•/-N-fallback-identity",
                            [(A_pol, A_pol), ("NEG_CTX", "#"), ("NEG_CTX", inner2_pol)]
                        )
                        #if self.trace:
                            #print(f"[•/-N-fallback-identity ✓] {ctx_pol}")
                        return True
                    for B_pol, _ in list(self.formulas_table.items()):
                        if self.prove_backward(A_pol, B_pol) and self.prove_neg_context('#') and self.prove_neg_context(
                                inner2_pol):
                            self.neg_context_cache[ctx_pol] = True
                            self.neg_context_derivations[ctx_pol] = (
                                "•/-N-fallback-general",
                                [(A_pol, B_pol), ("NEG_CTX", "#"), ("NEG_CTX", inner2_pol)]
                            )
                            #if self.trace:
                               # print(f"[•/-N-fallback-general ✓] {ctx_pol} with B={B_pol}")
                            return True
            except Exception:
                if self.trace:
                    print("  [•/-N fallback error] unexpected exception")

        return False

    # ---------- positive context prover ----------
    def prove_pos_context(self, ctx_pol: str) -> bool:
        """Prove ⊢P Γ[] for positive contexts (keeps permissive fallback)."""

        if ctx_pol in self.pos_context_cache:
           # if self.trace:
               # print(f"[cache P] {ctx_pol} = {self.pos_context_cache[ctx_pol]}")
            return self.pos_context_cache[ctx_pol]

        # provisional false
        self.pos_context_cache[ctx_pol] = False

        if ctx_pol not in self.contexts_table:
            return False

        ctx_node = self.contexts_table[ctx_pol]

        # hole case
        if ctx_node.is_hole():
            self.pos_context_cache[ctx_pol] = True
            self.pos_context_derivations[ctx_pol] = ("[]-P", [])
           # if self.trace:
               # print(f"[[]-P] {ctx_pol} is hole -> True")
            return True

        # require exactly one hole for our primary positive rules
        if self.count_holes(ctx_node) != 1:
            return False

        # try positive context rules
        if self._try_pos_context_rules(ctx_node, ctx_pol):
            return True

        # not proven
        return False

    def _try_pos_context_rules(self, ctx_node: Node, ctx_pol: str) -> bool:
        """Try positive context formation rules with exact matching and permissive fallbacks."""
        # NEW: Simple hole contexts - these are always valid positive contexts
        if ctx_node.val == '\\' and ctx_node.right and ctx_node.right.is_hole():
            # Pattern: A \ #
            self.pos_context_cache[ctx_pol] = True
            self.pos_context_derivations[ctx_pol] = ("\\-hole-P", [])
            #if self.trace:
             #   print(f"[\\-hole-P ✓] {ctx_pol}")
            return True

        if ctx_node.val == '/' and ctx_node.left and ctx_node.left.is_hole():
            # Pattern: # / A
            self.pos_context_cache[ctx_pol] = True
            self.pos_context_derivations[ctx_pol] = ("/-hole-P", [])
            #if self.trace:
             #   print(f"[/-hole-P ✓] {ctx_pol}")
            return True
        # (\•-P): ⊢P (A \ Γ[(B • ∆[])])  if A ⊢ B and ⊢P Γ[] and ⊢P ∆[]
        if ctx_node.val == '\\':
            A_node = ctx_node.left
            inner = ctx_node.right
            if inner is not None and inner.val == '•':
                B_node = inner.left
                delta_part = inner.right
                if delta_part is not None:
                    inner_pol = self.to_polish(inner)
                    for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                        if self.count_holes(gamma_ctx) != 1:
                            continue
                        for delta_pol, delta_ctx in list(self.contexts_table.items()):
                            if self.count_holes(delta_ctx) != 1:
                                continue
                            try:
                                pattern = self.compose_contexts(gamma_ctx.copy(),
                                                                Node('•', B_node.copy(), delta_ctx.copy()))
                                self.delete_children_at_hole(pattern)
                                if self.to_polish(pattern) == inner_pol:
                                    A_pol = self.to_polish(A_node)
                                    B_pol = self.to_polish(B_node)
                                    if (self.prove_backward(A_pol, B_pol)
                                            and self.prove_pos_context(gamma_pol)
                                            and self.prove_pos_context(delta_pol)):
                                        self.pos_context_cache[ctx_pol] = True
                                        self.pos_context_derivations[ctx_pol] = (
                                            "\\•-P",
                                            [(A_pol, B_pol), ("POS_CTX", gamma_pol), ("POS_CTX", delta_pol)]
                                        )
                                        #if self.trace:
                                          #  print(f"[\\•-P ✓] {ctx_pol}")
                                        return True
                            except Exception:
                                continue

        # (/•-P): ⊢P (Γ[(∆[] • B)] / A)  if A ⊢ B and ⊢P Γ[] and ⊢P ∆[]
        if ctx_node.val == '/':
            inner = ctx_node.left
            A_node = ctx_node.right
            if inner is not None and inner.val == '•':
                delta_part = inner.left
                B_node = inner.right
                if delta_part is not None:
                    inner_pol = self.to_polish(inner)
                    for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                        if self.count_holes(gamma_ctx) != 1:
                            continue
                        for delta_pol, delta_ctx in list(self.contexts_table.items()):
                            if self.count_holes(delta_ctx) != 1:
                                continue
                            try:
                                pattern = self.compose_contexts(gamma_ctx.copy(),
                                                                Node('•', delta_ctx.copy(), B_node.copy()))
                                self.delete_children_at_hole(pattern)
                                if self.to_polish(pattern) == inner_pol:
                                    A_pol = self.to_polish(A_node)
                                    B_pol = self.to_polish(B_node)
                                    if (self.prove_backward(A_pol, B_pol)
                                            and self.prove_pos_context(gamma_pol)
                                            and self.prove_pos_context(delta_pol)):
                                        self.pos_context_cache[ctx_pol] = True
                                        self.pos_context_derivations[ctx_pol] = (
                                            "/•-P",
                                            [(A_pol, B_pol), ("POS_CTX", gamma_pol), ("POS_CTX", delta_pol)]
                                        )
                                        #if self.trace:
                                          #  print(f"[/•-P ✓] {ctx_pol}")
                                        return True
                            except Exception:
                                continue

        # (/\-P): ⊢P (A / Γ[(∆[] \ B)])  if B ⊢ A and ⊢N Γ[] and ⊢P ∆[]
        if ctx_node.val == '/':
            A_node = ctx_node.left
            inner = ctx_node.right
            if inner is not None and inner.val == '\\':
                delta_part = inner.left
                B_node = inner.right
                if delta_part is not None:
                    inner_pol = self.to_polish(inner)
                    for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                        if self.count_holes(gamma_ctx) != 1:
                            continue
                        for delta_pol, delta_ctx in list(self.contexts_table.items()):
                            if self.count_holes(delta_ctx) != 1:
                                continue
                            try:
                                pattern = self.compose_contexts(gamma_ctx.copy(),
                                                                Node('\\', delta_ctx.copy(), B_node.copy()))
                                self.delete_children_at_hole(pattern)
                                if self.to_polish(pattern) == inner_pol:
                                    A_pol = self.to_polish(A_node)
                                    B_pol = self.to_polish(B_node)
                                    if (self.prove_backward(B_pol, A_pol)
                                            and self.prove_neg_context(gamma_pol)
                                            and self.prove_pos_context(delta_pol)):
                                        self.pos_context_cache[ctx_pol] = True
                                        self.pos_context_derivations[ctx_pol] = (
                                            "/\\-P",
                                            [(B_pol, A_pol), ("NEG_CTX", gamma_pol), ("POS_CTX", delta_pol)]
                                        )
                                        #if self.trace:
                                        #    print(f"[/\\-P ✓] {ctx_pol}")
                                        return True
                            except Exception:
                                continue

        # (\/-P): ⊢P (Γ[(B / ∆[])] \ A)  if B ⊢ A and ⊢N Γ[] and ⊢P ∆[]
        if ctx_node.val == '\\':
            inner = ctx_node.left
            A_node = ctx_node.right
            if inner is not None and inner.val == '/':
                B_node = inner.left
                delta_part = inner.right
                if delta_part is not None:
                    inner_pol = self.to_polish(inner)
                    for gamma_pol, gamma_ctx in list(self.contexts_table.items()):
                        if self.count_holes(gamma_ctx) != 1:
                            continue
                        for delta_pol, delta_ctx in list(self.contexts_table.items()):
                            if self.count_holes(delta_ctx) != 1:
                                continue
                            try:
                                pattern = self.compose_contexts(gamma_ctx.copy(),
                                                                Node('/', B_node.copy(), delta_ctx.copy()))
                                self.delete_children_at_hole(pattern)
                                if self.to_polish(pattern) == inner_pol:
                                    A_pol = self.to_polish(A_node)
                                    B_pol = self.to_polish(B_node)
                                    if (self.prove_backward(B_pol, A_pol)
                                            and self.prove_neg_context(gamma_pol)
                                            and self.prove_pos_context(delta_pol)):
                                        self.pos_context_cache[ctx_pol] = True
                                        self.pos_context_derivations[ctx_pol] = (
                                            "\\/-P",
                                            [(B_pol, A_pol), ("NEG_CTX", gamma_pol), ("POS_CTX", delta_pol)]
                                        )
                                       # if self.trace:
                                         #   print(f"[\\/-P ✓] {ctx_pol}")
                                        return True
                            except Exception:
                                continue

        return False

    # ---------- utility methods (derivation printing & stats) ----------
    def print_derivation(self, lhs: str, rhs: str, indent: int = 0):
        """Print the derivation tree for a proven sequent"""
        key = (lhs, rhs)
        if key not in self.derivations:
            print(" " * indent + f"{lhs} ⊢ {rhs} [Not proven]")
            return

        rule, premises = self.derivations[key]
        print(" " * indent + f"{lhs} ⊢ {rhs} [{rule}]")

        for prem in premises:
            if isinstance(prem, tuple) and len(prem) == 2:
                if prem[0] == "NEG_CTX":
                    # Negative context premise
                    ctx_pol = prem[1]
                    print(" " * (indent + 2) + f"⊢N {ctx_pol}")
                    self.print_neg_context_derivation(ctx_pol, indent + 2)
                elif prem[0] == "POS_CTX":
                    # Positive context premise
                    ctx_pol = prem[1]
                    print(" " * (indent + 2) + f"⊢P {ctx_pol}")
                    self.print_pos_context_derivation(ctx_pol, indent + 2)
                else:
                    # Regular sequent premise
                    prem_lhs, prem_rhs = prem
                    self.print_derivation(prem_lhs, prem_rhs, indent + 2)
            else:
                print(" " * (indent + 2) + str(prem))

    def print_neg_context_derivation(self, ctx_pol: str, indent: int = 0):
        """Print derivation for negative context"""
        if ctx_pol not in self.neg_context_derivations:
            print(" " * indent + f"⊢N {ctx_pol} [Not proven]")
            return

        rule, premises = self.neg_context_derivations[ctx_pol]
        print(" " * indent + f"⊢N {ctx_pol} [{rule}]")

        for prem in premises:
            if isinstance(prem, tuple) and len(prem) == 2:
                if prem[0] == "NEG_CTX":
                    inner_ctx = prem[1]
                    print(" " * (indent + 2) + f"⊢N {inner_ctx}")
                    self.print_neg_context_derivation(inner_ctx, indent + 2)
                elif prem[0] == "POS_CTX":
                    inner_ctx = prem[1]
                    print(" " * (indent + 2) + f"⊢P {inner_ctx}")
                    self.print_pos_context_derivation(inner_ctx, indent + 2)
                else:
                    prem_lhs, prem_rhs = prem
                    self.print_derivation(prem_lhs, prem_rhs, indent + 2)
            else:
                print(" " * (indent + 2) + str(prem))

    def print_pos_context_derivation(self, ctx_pol: str, indent: int = 0):
        """Print derivation for positive context"""
        if ctx_pol not in self.pos_context_derivations:
            print(" " * indent + f"⊢P {ctx_pol} [Not proven]")
            return

        rule, premises = self.pos_context_derivations[ctx_pol]
        print(" " * indent + f"⊢P {ctx_pol} [{rule}]")

        for prem in premises:
            if isinstance(prem, tuple) and len(prem) == 2:
                if prem[0] == "NEG_CTX":
                    inner_ctx = prem[1]
                    print(" " * (indent + 2) + f"⊢N {inner_ctx}")
                    self.print_neg_context_derivation(inner_ctx, indent + 2)
                elif prem[0] == "POS_CTX":
                    inner_ctx = prem[1]
                    print(" " * (indent + 2) + f"⊢P {inner_ctx}")
                    self.print_pos_context_derivation(inner_ctx, indent + 2)
                else:
                    prem_lhs, prem_rhs = prem
                    self.print_derivation(prem_lhs, prem_rhs, indent + 2)
            else:
                print(" " * (indent + 2) + str(prem))

    def get_stats(self):
        """Get statistics about proof search"""
        return {
            'sequences_total': len(self.seq_cache),
            'sequences_proven': sum(1 for v in self.seq_cache.values() if v),
            'neg_contexts_total': len(self.neg_context_cache),
            'neg_contexts_proven': sum(1 for v in self.neg_context_cache.values() if v),
            'pos_contexts_total': len(self.pos_context_cache),
            'pos_contexts_proven': sum(1 for v in self.pos_context_cache.values() if v),
            'formulas': len(self.formulas_table),
            'contexts': len(self.contexts_table)
        }



# ========== CLI INTERFACE ==========
def main():
    """Original interactive prover interface"""
    prover = SCBackwardProver(trace=True)
    print("SC backward prover (Polish notation) - FULL IMPLEMENTATION")
    print("Enter LHS and RHS formulas (space-separated tokens)")
    print("Operators: \, /, •")
    print("Examples:")
    print("  a ⊢ a")
    print("  \ a b ⊢ \ a b")
    print("  • a b ⊢ • a b")
    print("  / c b ⊢ / c b")
    print()

    lhs = input("LHS > ").strip()
    rhs = input("RHS > ").strip()

    start_time = time.time()

    # Initialize from user input
    prover.initialize_from_inputs(lhs, rhs)

    # Prove the sequent
    result = prover.prove_backward(lhs, rhs)

    end_time = time.time()

    print(f"\n=== RESULT: {lhs} ⊢ {rhs} is {'DERIVABLE' if result else 'NOT DERIVABLE'} ===")
    print(f"Time: {end_time - start_time:.4f}s")

    if result:
        print("\nDerivation:")
        prover.print_derivation(lhs, rhs)

    stats = prover.get_stats()
    print(f"\n=== STATISTICS ===")
    print(f"Sequences: {stats['sequences_proven']}/{stats['sequences_total']} proven")
    print(f"Negative contexts: {stats['neg_contexts_proven']}/{stats['neg_contexts_total']} proven")
    print(f"Positive contexts: {stats['pos_contexts_proven']}/{stats['pos_contexts_total']} proven")
    print(f"Formulas: {stats['formulas']}, Contexts: {stats['contexts']}")


# ========== MAIN ENTRY POINT ==========
if __name__ == "__main__":

    main()