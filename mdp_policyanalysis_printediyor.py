# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 22:59:32 2026

@author: birolduru
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, Tuple, List, Iterable, Optional
import math

Branch = str
State = Tuple[int, int, int, int]  # (xA0, xA1, xB0, xB1) for n=1

@dataclass(frozen=True)
class Costs:
    # Replenishment
    K0_A: float = 2.0
    K0_B: float = 2.0
    c0_A: float = 1.0
    c0_B: float = 1.0

    # Transshipment
    KT_A: float = 0.0   # fixed cost if A ships (A->B)
    KT_B: float = 0.0   # fixed cost if B ships (B->A)
    cT_A: float = 0.0   # per unit if A ships to B
    cT_B: float = 0.0   # per unit if B ships to A

    # Lost sales
    p_A: float = 5.0
    p_B: float = 5.0

    # Holding & waste
    h_A: float = 0.1
    h_B: float = 0.1
    w_A: float = 2.0
    w_B: float = 2.0

@dataclass
class Params:
    n: int = 1               # max age index; n=1 => ages 0 and 1
    Imax: int = 3            # max inventory per age per branch (toy)
    Qmax: int = 2            # max replenishment per branch
    Ymax: int = 2            # max transship per age (optional cap)

    # Demand pmfs for D1 and D2 separately, per branch.
    # Example: {0:0.5, 1:0.3, 2:0.2}
    D1_pmf_A: Dict[int, float] = None
    D1_pmf_B: Dict[int, float] = None
    D2_pmf_A: Dict[int, float] = None
    D2_pmf_B: Dict[int, float] = None

    def __post_init__(self):
        if self.D1_pmf_A is None:
            self.D1_pmf_A = {0: 0.4, 1: 0.4, 2: 0.2}
        if self.D1_pmf_B is None:
            self.D1_pmf_B = {0: 0.4, 1: 0.4, 2: 0.2}
        if self.D2_pmf_A is None:
            self.D2_pmf_A = {0: 0.5, 1: 0.3, 2: 0.2}
        if self.D2_pmf_B is None:
            self.D2_pmf_B = {0: 0.5, 1: 0.3, 2: 0.2}

class PerishableTwoBranchMDP:
    """
    Implements:
      - State s = (x^A_0..x^A_n, x^B_0..x^B_n)
      - Replenishment q=(qA,qB) adds to age0
      - FIFO issuing operator Γ(x,d) (oldest first)
      - Mid-period transshipment y age-structured, with unidirectional constraint
      - End-of-day: issue D2, then aging/outdating to next morning state
      - Costs: replenishment, transshipment, lost sales (part1, part2), holding, waste
      - RVI on the induced average-cost MDP using the two-epoch Bellman structure
    """

    def __init__(self, params: Params, costs: Costs):
        self.P = params
        self.C = costs
        assert self.P.n == 1, "Toy code currently set for n=1 (ages 0 and 1)."

        self.states = self._enumerate_states()
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.ref_state = (0, 0, 0, 0)

        # Precompute demand scenarios (D1, D2)
        self.D1_scenarios = self._joint_pmf(self.P.D1_pmf_A, self.P.D1_pmf_B)  # (dA, dB)->prob
        self.D2_scenarios = self._joint_pmf(self.P.D2_pmf_A, self.P.D2_pmf_B)

        # Precompute feasible replenishment actions
        self.repl_actions = list(product(range(self.P.Qmax + 1), repeat=2))  # (qA,qB)

    # ---------- utilities ----------
    def _enumerate_states(self) -> List[State]:
        I = range(self.P.Imax + 1)
        return [(a0, a1, b0, b1) for a0, a1, b0, b1 in product(I, I, I, I)]

    @staticmethod
    def _joint_pmf(pmf_A: Dict[int, float], pmf_B: Dict[int, float]) -> Dict[Tuple[int, int], float]:
        out = {}
        for dA, pA in pmf_A.items():
            for dB, pB in pmf_B.items():
                out[(dA, dB)] = pA * pB
        return out

    # ---------- FIFO issuing operator ----------
    @staticmethod
    def fifo_issue(x0: int, x1: int, d: int) -> Tuple[int, int, int]:
        """
        Oldest-first: use age1 first, then age0.
        Returns (rem0, rem1, lost).
        """
        use1 = min(x1, d)
        d_left = d - use1
        use0 = min(x0, d_left)
        lost = d_left - use0
        return (x0 - use0, x1 - use1, lost)

    # ---------- transitions ----------
    def after_replenishment(self, s: State, q: Tuple[int, int]) -> State:
        xA0, xA1, xB0, xB1 = s
        qA, qB = q
        # add to age0
        xA0b = min(self.P.Imax, xA0 + qA)
        xB0b = min(self.P.Imax, xB0 + qB)
        return (xA0b, xA1, xB0b, xB1)

    def mid_state_after_D1(self, s: State, q: Tuple[int, int], D1: Tuple[int, int]) -> Tuple[State, Tuple[int, int]]:
        """
        Applies replenishment then FIFO issue for D1.
        Returns (tilde_state, lost_sales=(lostA1, lostB1)).
        """
        xbar = self.after_replenishment(s, q)
        xA0, xA1, xB0, xB1 = xbar
        dA1, dB1 = D1
        rA0, rA1, lostA = self.fifo_issue(xA0, xA1, dA1)
        rB0, rB1, lostB = self.fifo_issue(xB0, xB1, dB1)
        return (rA0, rA1, rB0, rB1), (lostA, lostB)

    def feasible_transship_actions(self, tilde: State) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        y = (yAtoB, yBtoA), each is (y0, y1) shipped by age.
        Unidirectional: either A->B only, or B->A only, or none.
        Also bounded by available inventory at origin per age.
        """
        xA0, xA1, xB0, xB1 = tilde
        acts = []

        # None
        acts.append(((0, 0), (0, 0)))

        # A -> B only
        for y0 in range(min(xA0, self.P.Ymax) + 1):
            for y1 in range(min(xA1, self.P.Ymax) + 1):
                if y0 + y1 == 0:
                    continue
                acts.append(((y0, y1), (0, 0)))

        # B -> A only
        for y0 in range(min(xB0, self.P.Ymax) + 1):
            for y1 in range(min(xB1, self.P.Ymax) + 1):
                if y0 + y1 == 0:
                    continue
                acts.append(((0, 0), (y0, y1)))

        return acts

    def apply_transshipment(self, tilde: State, y: Tuple[Tuple[int, int], Tuple[int, int]]) -> State:
        (yA0, yA1), (yB0, yB1) = y
        xA0, xA1, xB0, xB1 = tilde

        # xhat^A = xtilde^A - yA->B + yB->A (agewise)
        xhatA0 = xA0 - yA0 + yB0
        xhatA1 = xA1 - yA1 + yB1
        xhatB0 = xB0 - yB0 + yA0
        xhatB1 = xB1 - yB1 + yA1

        # clip (should already be feasible)
        xhatA0 = max(0, min(self.P.Imax, xhatA0))
        xhatA1 = max(0, min(self.P.Imax, xhatA1))
        xhatB0 = max(0, min(self.P.Imax, xhatB0))
        xhatB1 = max(0, min(self.P.Imax, xhatB1))
        return (xhatA0, xhatA1, xhatB0, xhatB1)

    def end_of_day_and_next_state(self, xhat: State, D2: Tuple[int, int]) -> Tuple[State, Tuple[int, int], State]:
        """
        Issue D2 from xhat -> xstar (end-of-day remaining).
        Then aging/outdating to next morning state s'.
        Returns: (xstar, lost2, s_next)
        """
        xA0, xA1, xB0, xB1 = xhat
        dA2, dB2 = D2

        rA0, rA1, lostA2 = self.fifo_issue(xA0, xA1, dA2)
        rB0, rB1, lostB2 = self.fifo_issue(xB0, xB1, dB2)
        xstar = (rA0, rA1, rB0, rB1)

        # aging/outdating: for n=1 -> x0' = 0, x1' = x0*
        # items in age1* are the oldest and will be wasted (waste cost uses x1*)
        s_next = (0, rA0, 0, rB0)
        return xstar, (lostA2, lostB2), s_next
    
    def debug_one_day(
    self,
    s: State,
    q: Tuple[int, int],
    D1: Tuple[int, int],
    y: Tuple[Tuple[int, int], Tuple[int, int]],
    D2: Tuple[int, int],
) -> None:
        """
        Prints one-day trajectory + cost breakdown.
        """
        def fmt_state(st: State) -> str:
            xA0, xA1, xB0, xB1 = st
            return f"A:(0={xA0},1={xA1})  B:(0={xB0},1={xB1})"
    
        print("\n" + "=" * 80)
        print("DEBUG ONE DAY")
        print("- Input")
        print(f"  s   = {s}   | {fmt_state(s)}")
        print(f"  q   = {q}")
        print(f"  D1  = {D1}")
        print(f"  y   = {y}    (yAtoB, yBtoA)")
        print(f"  D2  = {D2}")
    
        # Replenishment
        xbar = self.after_replenishment(s, q)
        c_repl = self.cost_replenishment(q)
        print("\n- After replenishment (xbar)")
        print(f"  xbar = {xbar} | {fmt_state(xbar)}")
        print(f"  Replenishment cost = {c_repl:.4f}")
    
        # Issue D1
        tilde, lost1 = self.mid_state_after_D1(s, q, D1)
        c_ls1 = self.cost_lost_sales_part1(lost1)
        print("\n- After issuing D1 (tilde)")
        print(f"  tilde = {tilde} | {fmt_state(tilde)}")
        print(f"  lost1 = {lost1} -> cost = {c_ls1:.4f}")
    
        # Feasibility check for y
        (yA0, yA1), (yB0, yB1) = y
        xA0, xA1, xB0, xB1 = tilde
        feasible = True
        if not (0 <= yA0 <= xA0 and 0 <= yA1 <= xA1 and 0 <= yB0 <= xB0 and 0 <= yB1 <= xB1):
            feasible = False
        if (yA0 + yA1) > 0 and (yB0 + yB1) > 0:
            feasible = False
    
        print("\n- Transshipment feasibility")
        print(f"  feasible? {feasible}")
        if not feasible:
            print("  WARNING: y violates feasibility (bounds and/or unidirectional).")
    
        # Apply transshipment
        xhat = self.apply_transshipment(tilde, y)
        c_tr = self.cost_transshipment(y)
        print("\n- After transshipment (xhat)")
        print(f"  xhat = {xhat} | {fmt_state(xhat)}")
        print(f"  Transshipment cost = {c_tr:.4f}")
    
        # Issue D2 + next state
        xstar, lost2, s_next = self.end_of_day_and_next_state(xhat, D2)
        c_ls2 = self.cost_lost_sales_part2(lost2)
        c_hw = self.cost_holding_and_waste(xstar)
    
        print("\n- After issuing D2 (xstar)")
        print(f"  xstar = {xstar} | {fmt_state(xstar)}")
        print(f"  lost2 = {lost2} -> cost = {c_ls2:.4f}")
    
        rA0, rA1, rB0, rB1 = xstar
        holding = self.C.h_A * rA0 + self.C.h_B * rB0
        waste = self.C.w_A * rA1 + self.C.w_B * rB1
        print("\n- Holding & waste")
        print(f"  holding = {holding:.4f}, waste = {waste:.4f}, total = {c_hw:.4f}")
    
        print("\n- Next period state")
        print(f"  s_next = {s_next} | {fmt_state(s_next)}")
    
        total = c_repl + c_ls1 + c_tr + c_ls2 + c_hw
        print("\n- Total scenario cost")
        print(f"  total = {total:.4f}")
        print("=" * 80 + "\n")


    # ---------- costs ----------
    def cost_replenishment(self, q: Tuple[int, int]) -> float:
        qA, qB = q
        cost = 0.0
        if qA > 0:
            cost += self.C.K0_A + self.C.c0_A * qA
        if qB > 0:
            cost += self.C.K0_B + self.C.c0_B * qB
        return cost

    def cost_lost_sales_part1(self, lost1: Tuple[int, int]) -> float:
        lostA, lostB = lost1
        return self.C.p_A * lostA + self.C.p_B * lostB

    def cost_transshipment(self, y: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        (yA0, yA1), (yB0, yB1) = y
        cost = 0.0
        shipA = yA0 + yA1
        shipB = yB0 + yB1
        if shipA > 0:
            cost += self.C.KT_A + self.C.cT_A * shipA
        if shipB > 0:
            cost += self.C.KT_B + self.C.cT_B * shipB
        return cost

    def cost_lost_sales_part2(self, lost2: Tuple[int, int]) -> float:
        lostA, lostB = lost2
        return self.C.p_A * lostA + self.C.p_B * lostB

    def cost_holding_and_waste(self, xstar: State) -> float:
        # holding for k=0..n-1 => only age0 when n=1
        rA0, rA1, rB0, rB1 = xstar
        holding = self.C.h_A * rA0 + self.C.h_B * rB0
        waste = self.C.w_A * rA1 + self.C.w_B * rB1
        return holding + waste

    # ---------- Bellman pieces ----------
    def compute_mid_value(self, tilde: State, V_next: Dict[State, float]) -> float:
        """
        V(tilde) = min_y { cT(y) + E_D2[ cL2 + cH+cW + V(s') ] }
        """
        best = float("inf")
        for y in self.feasible_transship_actions(tilde):
            cT = self.cost_transshipment(y)
            xhat = self.apply_transshipment(tilde, y)

            exp = 0.0
            for D2, prob in self.D2_scenarios.items():
                xstar, lost2, s_next = self.end_of_day_and_next_state(xhat, D2)
                c = self.cost_lost_sales_part2(lost2) + self.cost_holding_and_waste(xstar) + V_next[s_next]
                exp += prob * c

            total = cT + exp
            
            if total < best:
                best = total
        return best

    def compute_start_value(self, s: State, V_next: Dict[State, float]) -> float:
        """
        RHS(s) = min_q { c0(q) + E_D1[ cL1 + V(tilde) ] }
        """
        best = float("inf")
        for q in self.repl_actions:
            c0 = self.cost_replenishment(q)
            exp = 0.0
            for D1, prob in self.D1_scenarios.items():
                tilde, lost1 = self.mid_state_after_D1(s, q, D1)
                c = self.cost_lost_sales_part1(lost1) + self.compute_mid_value(tilde, V_next)
                exp += prob * c
            total = c0 + exp
            
            if total < best:
                
                best = total
        return best

    # ---------- Relative Value Iteration ----------
    def solve_rvi(self, max_iter: int = 300, tol: float = 1e-2, verbose: bool = True):
        """
        RVI update:
          T(V)(s) = RHS(s)   (two-epoch minimization already inside RHS)
          Normalize: V_new(s) = T(V_old)(s) - T(V_old)(s_ref)
        Gain estimate: lambda ≈ T(V_old)(s_ref)
        """
        V = {s: 0.0 for s in self.states}
        lam = 0.0

        for it in range(1, max_iter + 1):
            T_V = {}
            for s in self.states:
                T_V[s] = self.compute_start_value(s, V)

            lam_new = T_V[self.ref_state]
            V_new = {s: (T_V[s] - lam_new) for s in self.states}  # so V_new(ref)=0

            # sup norm diff
            diff = max(abs(V_new[s] - V[s]) for s in self.states)
            V = V_new
            lam = lam_new

            if verbose and (it % 10 == 0 or it == 1):
                print(f"iter={it:4d}  lambda~{lam:.6f}  diff={diff:.3e}")

            if diff < tol:
                if verbose:
                    print(f"Converged at iter={it}, lambda~{lam:.6f}")
                break

        return V, lam

    # ---------- policy extraction (optional) ----------
    def greedy_policy(self, V: Dict[State, float]):
        """
        Returns greedy replenishment and transshipment decisions under V.
        """
        pol_q = {}
        pol_y = {}

        # Precompute mid-value argmin y for each tilde if desired
        for s in self.states:
            # best q
            best_q = None
            best_val = float("inf")
            for q in self.repl_actions:
                c0 = self.cost_replenishment(q)
                exp = 0.0
                for D1, prob in self.D1_scenarios.items():
                    tilde, lost1 = self.mid_state_after_D1(s, q, D1)
                    c = self.cost_lost_sales_part1(lost1) + self.compute_mid_value(tilde, V)
                    exp += prob * c
                total = c0 + exp
                if total < best_val:
                    best_val = total
                    best_q = q
            pol_q[s] = best_q

        # transshipment policy is defined at mid states, so we can compute on the fly:
        # here just return a function
        def best_y_for_tilde(tilde: State) -> Tuple[Tuple[int, int], Tuple[int, int]]:
            best_y = ((0,0),(0,0))
            best = float("inf")
            for y in self.feasible_transship_actions(tilde):
                cT = self.cost_transshipment(y)
                xhat = self.apply_transshipment(tilde, y)
                exp = 0.0
                for D2, prob in self.D2_scenarios.items():
                    xstar, lost2, s_next = self.end_of_day_and_next_state(xhat, D2)
                    c = self.cost_lost_sales_part2(lost2) + self.cost_holding_and_waste(xstar) + V[s_next]
                    exp += prob * c
                total = cT + exp
                if total < best:
                    best = total
                    best_y = y
            return best_y

        return pol_q, best_y_for_tilde
        
    def verify_replenishment_verbose(self, s, V):
        print(f"\n{'='*30} İKMAL (REPLENISHMENT) ANALİZİ {'='*30}")
        print(f"Durum (s): {s} (A0, A1, B0, B1)")
        print(f"{'Aksiyon (qA, qB)':<20} | {'Sipariş Mal.':<12} | {'Beklenen Gel. Mal.':<20} | {'Toplam Maliyet'}")
        print("-" * 85)
        best_q, min_total = None, float('inf')
        for q in self.repl_actions:
            c0 = self.cost_replenishment(q)
            exp_future = 0.0
            for D1, prob in self.D1_scenarios.items():
                tilde, lost1 = self.mid_state_after_D1(s, q, D1)
                exp_future += prob * (self.cost_lost_sales_part1(lost1) + self.compute_mid_value(tilde, V))
            total = c0 + exp_future
            if total < min_total:
                best_q, min_total = q, total
            print(f"{str(q):<20} | {c0:<12.2f} | {exp_future:<20.4f} | {total:.6f}")
        print("-" * 85)
        print(f"==> OPTİMAL İKMAL KARARI: {best_q} (Maliyet: {min_total:.6f})")

    def verify_transshipment_verbose(self, tilde, V):
        print(f"\n{'='*30} TRANSFER (TRANSSHIPMENT) ANALİZİ {'='*30}")
        print(f"Gün Ortası Durumu (tilde): {tilde} (A0, A1, B0, B1)")
        print(f"{'Aksiyon (yAtoB, yBtoA)':<25} | {'Trans. Mal.':<12} | {'Beklenen Gel. Mal.':<20} | {'Toplam Maliyet'}")
        print("-" * 90)
        best_y, min_total = None, float('inf')
        for y in self.feasible_transship_actions(tilde):
            cT = self.cost_transshipment(y)
            xhat = self.apply_transshipment(tilde, y)
            exp_future = 0.0
            for D2, prob in self.D2_scenarios.items():
                xstar, lost2, s_next = self.end_of_day_and_next_state(xhat, D2)
                exp_future += prob * (self.cost_lost_sales_part2(lost2) + self.cost_holding_and_waste(xstar) + V[s_next])
            total = cT + exp_future
            if total < min_total:
                best_y, min_total = y, total
            print(f"{str(y):<25} | {cT:<12.2f} | {exp_future:<20.4f} | {total:.6f}")
        print("-" * 90)
        print(f"==> OPTİMAL TRANSFER KARARI: {best_y} (Maliyet: {min_total:.6f})")
        

if __name__ == "__main__":
    # 1. Parametreleri ve Maliyetleri tanımla
    P = Params(n=1, Imax=3, Qmax=2, Ymax=2) 
    C = Costs()
    mdp = PerishableTwoBranchMDP(P, C)
    
    # 2. Modeli çöz (V değerlerini bul)
    V, lam = mdp.solve_rvi(max_iter=200, tol=1e-2, verbose=True)

    # 3. İKMAL ANALİZİ (Örnek: Her yer boşken ne yapıyor?)
    test_s = (0, 0, 0, 0)
    mdp.verify_replenishment_verbose(test_s, V)

    # 4. TRANSFER ANALİZİ (Örnek: A boş, B'de 2 tane yeni ürün varken ne yapıyor?)
    # tilde durumu: sipariş verilmiş ve ilk talep (D1) karşılanmış haldeki stoktur.
    test_tilde = (0, 0, 2, 0) 
    mdp.verify_transshipment_verbose(test_tilde, V)