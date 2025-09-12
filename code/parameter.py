
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import networkx as nx
from typing import Callable, Optional, Tuple


def estimate_eps_kneedle(G: nx.Graph, 
                        similarity_func: Callable = None,
                        mu: int = 6, 
                        gamma: float = 0.5,
                        plot: bool = True,
                        curve: str = 'convex',
                        direction: str = 'decreasing',
                        sensitivity: float = 1.0) -> float:
    """
    가중치 그래프에서 각 노드의 mu번째 큰 similarity를 기반으로 
    kneedle point를 찾아 eps 값을 자동 추정합니다.
    
    Parameters:
    -----------
    G : nx.Graph
        입력 그래프 (가중치 정보 포함)
    similarity_func : Callable, optional
        유사도 계산 함수. None이면 기본 유사도 함수 사용
    mu : int, default=6
        각 노드에서 고려할 이웃의 수 (mu번째 큰 similarity)
    gamma : float, default=0.5
        kneedle 알고리즘의 민감도 파라미터 및 similarity 함수 파라미터
    plot : bool, default=True
        결과 시각화 여부
    curve : str, default='convex'
        kneedle 곡선 타입 ('concave' or 'convex')
    direction : str, default='decreasing'
        kneedle 방향 ('increasing' or 'decreasing')
    
    Returns:
    --------
    float
        추정된 eps 값
    """
    
    mu_similarities = []
    
    # 각 노드에 대해 mu번째 큰 similarity 계산
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) == 0:
            # 고립된 노드의 경우
            mu_similarities.append(0.0)
            continue
            
        # 모든 이웃과의 similarity 계산
        similarities = []
        for neighbor in neighbors:
            sim = similarity_func(G, node, neighbor, gamma)
            similarities.append(sim)
        
        # similarity 내림차순 정렬
        similarities.sort(reverse=True)
        
        # mu번째 큰 similarity 선택 (인덱스는 mu-1)
        if len(similarities) >= mu:
            mu_sim = similarities[mu-1]
        else:
            # 이웃이 mu개보다 적으면 가장 작은 similarity 사용
            mu_sim = similarities[-1] if similarities else 0.0
            
        mu_similarities.append(mu_sim)
    
    # mu번째 similarity들을 내림차순으로 정렬 (높은 값부터)
    sorted_similarities = sorted(mu_similarities, reverse=True)
    
    # kneedle point 찾기
    if len(sorted_similarities) < 3:
        # 데이터가 너무 적으면 중앙값 반환
        eps_estimate = np.median(sorted_similarities)
        print(f"Warning: 데이터가 부족하여 중앙값을 사용합니다. eps = {eps_estimate:.4f}")
        return eps_estimate
    
    x = np.arange(len(sorted_similarities))
    y = np.array(sorted_similarities)
    
    try:
        # 여러 방법으로 knee point 시도
        knee_candidates = []
        
        # 방법 1: 기본 kneedle (더 민감하게)
        try:
            kneedle1 = KneeLocator(x, y, 
                                 curve='convex', 
                                 direction='decreasing',
                                 S=sensitivity*0.3)  # 더 민감하게
            if kneedle1.knee is not None:
                knee_candidates.append(('kneedle_sensitive', kneedle1.knee))
        except:
            pass
            
        # 방법 2: concave curve 시도
        try:
            kneedle2 = KneeLocator(x, y, 
                                 curve='concave', 
                                 direction='decreasing',
                                 S=sensitivity*0.2)
            if kneedle2.knee is not None:
                knee_candidates.append(('kneedle_concave', kneedle2.knee))
        except:
            pass
        
        # 방법 3: 기울기 기반 방법 (가장 급격한 변화 지점)
        if len(y) > 3:
            gradients = np.diff(y)
            # 상위 75%를 제외하고 찾기 (너무 높은 similarity 제외)
            start_idx = max(1, int(len(gradients) * 0.1))  # 상위 10% 제외
            end_idx = min(len(gradients), int(len(gradients) * 0.8))  # 하위 20% 제외
            
            if end_idx > start_idx:
                search_gradients = gradients[start_idx:end_idx]
                steepest_relative_idx = np.argmin(search_gradients)
                steepest_idx = start_idx + steepest_relative_idx
                knee_candidates.append(('gradient_based', steepest_idx))
        
        # 방법 4: 분위수 기반 (중간 영역에서 급격한 변화)
        q40_idx = int(len(sorted_similarities) * 0.4)  # 상위 40% 이후부터
        q80_idx = int(len(sorted_similarities) * 0.8)  # 하위 20% 전까지
        
        if q80_idx > q40_idx + 2:
            mid_gradients = np.diff(y[q40_idx:q80_idx])
            if len(mid_gradients) > 0:
                mid_steepest_idx = np.argmin(mid_gradients)
                knee_candidates.append(('quantile_based', q40_idx + mid_steepest_idx))
        
        # 후보들 중 선택
        if knee_candidates:
            # 너무 앞쪽(높은 similarity)에 있는 후보들 필터링
            min_acceptable_idx = int(len(sorted_similarities) * 0.2)  # 최소 20% 지점 이후
            valid_candidates = [(name, idx) for name, idx in knee_candidates 
                              if idx >= min_acceptable_idx]
            
            if valid_candidates:
                # 가장 앞쪽의 유효한 후보 선택 (가장 보수적)
                knee_method, knee_point = min(valid_candidates, key=lambda x: x[1])
                eps_estimate = sorted_similarities[knee_point]
                # print(f"Knee point found using {knee_method} at index {knee_point}, eps = {eps_estimate:.4f}")
            else:
                # 모든 후보가 너무 앞쪽인 경우, 40% 지점 사용
                knee_point = int(len(sorted_similarities) * 0.4)
                eps_estimate = sorted_similarities[knee_point]
                print(f"Using 40% quantile as eps = {eps_estimate:.4f}")
        else:
            # 후보를 찾지 못한 경우
            knee_point = int(len(sorted_similarities) * 0.5)
            eps_estimate = sorted_similarities[knee_point]
            print(f"Fallback: Using median position, eps = {eps_estimate:.4f}")
            
    except Exception as e:
        print(f"Kneedle 계산 중 오류 발생: {e}")
        knee_point = int(len(sorted_similarities) * 0.5)
        eps_estimate = sorted_similarities[knee_point]
        print(f"Error fallback: eps = {eps_estimate:.4f}")
    
    # 시각화
    if plot:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', linewidth=2, label='Sorted μ-th similarities (descending)')
        if knee_point is not None:
            plt.axvline(x=knee_point, color='r', linestyle='--', 
                       label=f'Kneedle point (eps={eps_estimate:.4f})')
            plt.plot(knee_point, sorted_similarities[knee_point], 
                    'ro', markersize=8, label='Eps estimate')
        plt.xlabel('Node index (sorted by similarity desc)')
        plt.ylabel('μ-th largest similarity')
        plt.title(f'μ-th Similarities Distribution (μ={mu})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 기울기도 함께 표시 (여러 후보 지점들도 표시)
        if len(y) > 1:
            ax2 = plt.twinx()
            gradients = np.diff(y)
            ax2.plot(x[:-1], gradients, 'g--', alpha=0.5, label='Gradient')
            ax2.set_ylabel('Gradient', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # 분위수 영역 표시
            q20_idx = int(len(sorted_similarities) * 0.2)
            q40_idx = int(len(sorted_similarities) * 0.4)
            q80_idx = int(len(sorted_similarities) * 0.8)
            
            plt.axvspan(q20_idx, q40_idx, alpha=0.1, color='yellow', label='Search region 1')
            plt.axvspan(q40_idx, q80_idx, alpha=0.1, color='orange', label='Search region 2')
        
        plt.subplot(1, 2, 2)
        plt.hist(sorted_similarities, bins=min(30, len(sorted_similarities)//2), 
                alpha=0.7, edgecolor='black')
        plt.axvline(x=eps_estimate, color='r', linestyle='--', linewidth=2,
                   label=f'Eps estimate = {eps_estimate:.4f}')
        plt.xlabel('μ-th similarity value')
        plt.ylabel('Frequency')
        plt.title('Distribution of μ-th Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 정보 출력
        print(f"\n=== Eps Estimation Results ===")
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        print(f"μ parameter: {mu}")
        print(f"Estimated eps: {eps_estimate:.6f}")
        print(f"Min μ-th similarity: {min(sorted_similarities):.6f}")
        print(f"Max μ-th similarity: {max(sorted_similarities):.6f}")
        print(f"Mean μ-th similarity: {np.mean(sorted_similarities):.6f}")
        print(f"Median μ-th similarity: {np.median(sorted_similarities):.6f}")
    
    return eps_estimate


# 사용 예시

if __name__ == "__main__":

    import networkx as nx
    from utils import load_ground_truth
    import similarity

    network = "karate"
    dataset = "../dataset/real/" + network + "/network.dat"
    G = nx.read_weighted_edgelist(dataset, nodetype=int)
    gt_path = "../dataset/real/" + network + "/labels.dat"
    ground_truth = load_ground_truth(gt_path)

    eps= estimate_eps_kneedle(G, similarity_func=similarity.Gen_wscan_similarity, mu=4, gamma=0.5)
    print("estimated eps =", eps)