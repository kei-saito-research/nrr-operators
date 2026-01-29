"""
Non-Resolution Reasoning (NRR) Prototype Implementation
状態空間管理と8つの原理満足型オペレーター
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Interpretation:
    """単一の解釈"""
    semantic_vector: str  # 意味表現 (後でembeddingに置き換え可能)
    context: str          # 文脈識別子
    weight: float         # 活性化重み [0, 1]
    metadata: Dict = field(default_factory=dict)  # 追加情報


class NRRState:
    """NRR状態空間 S = {(v_i, c_i, w_i)}"""
    
    def __init__(self, interpretations: List[Interpretation]):
        self.interpretations = interpretations
    
    def get_weights(self) -> np.ndarray:
        """重み配列を取得"""
        return np.array([interp.weight for interp in self.interpretations])
    
    def entropy(self) -> float:
        """Shannon entropy H(S) を計算"""
        weights = self.get_weights()
        if weights.sum() == 0:
            return 0.0
        
        # 正規化して確率分布に
        probs = weights / weights.sum()
        
        # エントロピー計算 (0 log 0 = 0 と扱う)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def size(self) -> int:
        """解釈の数"""
        return len(self.interpretations)
    
    def __repr__(self):
        return f"NRRState({self.size()} interpretations, H={self.entropy():.3f})"


class NRROperators:
    """8つの原理満足型オペレーター"""
    
    @staticmethod
    def dampening(state: NRRState, lambda_param: float = 0.3) -> NRRState:
        """
        δ: Dampening - 平均への圧縮
        w'_i = w_i(1-λ) + w̄λ
        
        支配的な解釈を抑え、エントロピーを増加させる
        """
        weights = state.get_weights()
        w_mean = weights.mean()
        
        new_weights = weights * (1 - lambda_param) + w_mean * lambda_param
        
        new_interpretations = [
            Interpretation(
                semantic_vector=interp.semantic_vector,
                context=interp.context,
                weight=new_w,
                metadata=interp.metadata
            )
            for interp, new_w in zip(state.interpretations, new_weights)
        ]
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def stripping(state: NRRState, bias: float = 0.1) -> NRRState:
        """
        σ: Stripping - 比例的バイアス除去
        w'_i = w_i - b * (w_i / max(w))
        
        高い重みほど多く減少させるが、比率は保存
        """
        weights = state.get_weights()
        max_w = weights.max()
        
        if max_w == 0:
            return state
        
        new_weights = weights - bias * (weights / max_w)
        new_weights = np.maximum(new_weights, 0)  # 負にならないように
        
        new_interpretations = [
            Interpretation(
                semantic_vector=interp.semantic_vector,
                context=interp.context,
                weight=new_w,
                metadata=interp.metadata
            )
            for interp, new_w in zip(state.interpretations, new_weights)
        ]
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def positioning(state: NRRState, timestamp: Optional[int] = None) -> NRRState:
        """
        ρ: Positioning - 時系列座標の付与
        (v_i, c_i, w_i) → (v_i, (c_i, t), w_i)
        
        文脈に時間情報を追加
        """
        if timestamp is None:
            import time
            timestamp = int(time.time())
        
        new_interpretations = [
            Interpretation(
                semantic_vector=interp.semantic_vector,
                context=f"{interp.context}@t{timestamp}",
                weight=interp.weight,
                metadata={**interp.metadata, 'timestamp': timestamp}
            )
            for interp in state.interpretations
        ]
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def abstraction(state: NRRState) -> NRRState:
        """
        α: Abstraction - 関係構造の付与
        (v_i, c_i, w_i) → (v_i, c_i, w_i, R_i)
        
        解釈間の関係を計算してメタデータに追加
        """
        n = state.size()
        
        new_interpretations = []
        for i, interp in enumerate(state.interpretations):
            # 他の解釈との「距離」を記録（ここでは単純化）
            relations = {
                f'dist_to_{j}': abs(interp.weight - state.interpretations[j].weight)
                for j in range(n) if j != i
            }
            
            new_interpretations.append(
                Interpretation(
                    semantic_vector=interp.semantic_vector,
                    context=interp.context,
                    weight=interp.weight,
                    metadata={**interp.metadata, 'relations': relations}
                )
            )
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def invariance(state: NRRState, variance_threshold: float = 0.1) -> NRRState:
        """
        ι: Invariance - 安定構造の抽出
        文脈変化に対して安定な解釈のみ保持
        
        （簡易実装：重みが閾値以上のものを保持）
        """
        new_interpretations = [
            interp for interp in state.interpretations
            if interp.weight >= variance_threshold
        ]
        
        if not new_interpretations:
            # 全て除外された場合は元のまま
            return state
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def deferred_resolution(state: NRRState) -> NRRState:
        """
        τ: Deferred Resolution - 遅延決定
        τ(S) = S
        
        出力境界まで何もしない（恒等写像）
        """
        return state
    
    @staticmethod
    def cpp_integration(state1: NRRState, state2: NRRState) -> NRRState:
        """
        κ: CPP Integration - 矛盾保存統合
        κ(S, S') = S ∪ S' (矛盾タグ付き)
        
        2つの状態を矛盾を保存したまま統合
        """
        # 矛盾を検出（簡易版：同じcontext で異なるweightの解釈）
        context_map = {}
        for interp in state1.interpretations + state2.interpretations:
            if interp.context not in context_map:
                context_map[interp.context] = []
            context_map[interp.context].append(interp)
        
        new_interpretations = []
        for context, interps in context_map.items():
            if len(interps) > 1:
                # 矛盾を検出してタグ付け
                for idx, interp in enumerate(interps):
                    new_interpretations.append(
                        Interpretation(
                            semantic_vector=interp.semantic_vector,
                            context=interp.context,
                            weight=interp.weight,
                            metadata={**interp.metadata, 'conflict_group': context, 'variant': idx}
                        )
                    )
            else:
                new_interpretations.append(interps[0])
        
        return NRRState(new_interpretations)
    
    @staticmethod
    def persistence(state_current: NRRState, state_previous: NRRState, 
                   decay: float = 0.5) -> NRRState:
        """
        π: Persistence - 時系列持続
        π(S_t, S_{t-1}) = S_t ∪ {(v_i, c_i, λw_i) : (v_i, c_i, w_i) ∈ S_{t-1}}
        
        過去の解釈を減衰させて保持
        """
        # 現在の解釈
        new_interpretations = state_current.interpretations.copy()
        
        # 過去の解釈を減衰させて追加
        for interp in state_previous.interpretations:
            new_interpretations.append(
                Interpretation(
                    semantic_vector=interp.semantic_vector,
                    context=f"{interp.context}_prev",
                    weight=interp.weight * decay,
                    metadata={**interp.metadata, 'is_historical': True}
                )
            )
        
        return NRRState(new_interpretations)


class CollapseDetector:
    """崩壊検出器"""
    
    @staticmethod
    def detect_collapse(state_before: NRRState, state_after: NRRState, 
                       epsilon: float = 0.1) -> Tuple[bool, float]:
        """
        情報崩壊を検出
        
        Returns:
            (collapsed, delta_H): 崩壊したか、エントロピー変化量
        """
        h_before = state_before.entropy()
        h_after = state_after.entropy()
        delta_h = h_after - h_before
        
        collapsed = delta_h < -epsilon
        
        return collapsed, delta_h
    
    @staticmethod
    def information_loss(state_before: NRRState, state_after: NRRState) -> float:
        """
        情報損失量 L = H(S) - H(O(S))
        正の値は崩壊を示す
        """
        return state_before.entropy() - state_after.entropy()


def projection_for_output(state: NRRState) -> Interpretation:
    """
    Π: Non-Destructive Projection
    出力のために最大重みの解釈を選択（状態は破壊しない）
    """
    if not state.interpretations:
        raise ValueError("Empty state cannot produce output")
    
    return max(state.interpretations, key=lambda x: x.weight)


# =============================================================================
# LLM統合: 既存LLMの後処理層として動作
# =============================================================================

class LLMtoNRRBridge:
    """既存LLMの出力をNRR状態に変換"""
    
    @staticmethod
    def extract_interpretations_from_llm(llm_response: str) -> NRRState:
        """
        LLM出力から複数解釈を抽出
        
        想定フォーマット:
        1. [解釈1の説明] (confidence: 0.7)
        2. [解釈2の説明] (confidence: 0.2)
        3. [解釈3の説明] (confidence: 0.1)
        """
        interpretations = []
        
        # 簡易パーサー（実際はもっと堅牢に）
        lines = llm_response.strip().split('\n')
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            
            # confidence抽出
            import re
            confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if confidence_match:
                weight = float(confidence_match.group(1))
            else:
                weight = 1.0 / len(lines)  # デフォルト均等配分
            
            # 解釈テキスト抽出
            interpretation_text = re.sub(r'\(confidence[^)]*\)', '', line, flags=re.IGNORECASE).strip()
            interpretation_text = re.sub(r'^\d+\.\s*', '', interpretation_text)
            
            interpretations.append(
                Interpretation(
                    semantic_vector=interpretation_text,
                    context=f"llm_output_{idx}",
                    weight=weight
                )
            )
        
        return NRRState(interpretations)
    
    @staticmethod
    def format_state_for_llm(state: NRRState) -> str:
        """
        NRR状態をLLMが読める形式に変換
        """
        output = f"Current state entropy: {state.entropy():.3f} bits\n\n"
        output += f"{state.size()} active interpretations:\n\n"
        
        for idx, interp in enumerate(sorted(state.interpretations, 
                                           key=lambda x: x.weight, 
                                           reverse=True)):
            output += f"{idx+1}. {interp.semantic_vector}\n"
            output += f"   Weight: {interp.weight:.3f}, Context: {interp.context}\n"
            if interp.metadata:
                output += f"   Metadata: {interp.metadata}\n"
            output += "\n"
        
        return output


# =============================================================================
# 使用例
# =============================================================================

if __name__ == "__main__":
    print("=== NRR Prototype Demo ===\n")
    
    # 1. 状態空間の作成
    print("1. 初期状態の作成")
    initial_interpretations = [
        Interpretation("Financial collapse", "context_economic", 0.7),
        Interpretation("Personal dissolution", "context_psychological", 0.2),
        Interpretation("Transformative moment", "context_spiritual", 0.1),
    ]
    state = NRRState(initial_interpretations)
    print(f"   {state}")
    print(f"   Initial entropy: {state.entropy():.3f} bits\n")
    
    # 2. Dampeningオペレーターの適用
    print("2. Dampening (δ) を適用")
    state_dampened = NRROperators.dampening(state, lambda_param=0.3)
    print(f"   {state_dampened}")
    collapsed, delta_h = CollapseDetector.detect_collapse(state, state_dampened)
    print(f"   ΔH = {delta_h:+.3f} bits (collapsed: {collapsed})\n")
    
    # 3. 比較: 原理違反オペレーター（uniform subtraction）
    print("3. 原理違反: 一様減算 (w_i - 0.1)")
    weights = state.get_weights()
    bad_weights = weights - 0.1
    bad_weights = np.maximum(bad_weights, 0)
    bad_state = NRRState([
        Interpretation(interp.semantic_vector, interp.context, w)
        for interp, w in zip(state.interpretations, bad_weights)
    ])
    print(f"   {bad_state}")
    collapsed, delta_h = CollapseDetector.detect_collapse(state, bad_state)
    print(f"   ΔH = {delta_h:+.3f} bits (collapsed: {collapsed})\n")
    
    # 4. CPP Integration
    print("4. CPP Integration (κ)")
    new_interpretations = [
        Interpretation("Structural decomposition", "context_economic", 0.6),
        Interpretation("Creative destruction", "context_economic", 0.4),
    ]
    state2 = NRRState(new_interpretations)
    state_integrated = NRROperators.cpp_integration(state, state2)
    print(f"   {state_integrated}")
    print(f"   統合後のエントロピー: {state_integrated.entropy():.3f} bits\n")
    
    # 5. 出力投影
    print("5. 出力投影 (Π)")
    output = projection_for_output(state)
    print(f"   Selected for output: \"{output.semantic_vector}\"")
    print(f"   (内部状態は保持: {state})\n")
    
    # 6. LLM統合の例
    print("6. LLM統合の例")
    llm_output = """
1. The person is experiencing acute distress (confidence: 0.6)
2. External circumstances are changing rapidly (confidence: 0.3)
3. A spiritual transformation is occurring (confidence: 0.1)
    """
    state_from_llm = LLMtoNRRBridge.extract_interpretations_from_llm(llm_output)
    print(f"   {state_from_llm}")
    print(f"   LLM出力のエントロピー: {state_from_llm.entropy():.3f} bits\n")
    
    formatted = LLMtoNRRBridge.format_state_for_llm(state_from_llm)
    print("   NRR状態 → LLM入力形式:")
    print("   " + "\n   ".join(formatted.split('\n')[:5]) + "...\n")
    
    print("=== Demo Complete ===")
