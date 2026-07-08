import jax
import jax.numpy as jnp
import optax
import numpy as np

# ==========================================
# 1. DATA PREPARATION 
# ==========================================
np.random.seed(42)
N = 10 # Number of scenarios

D_features = np.random.uniform(0, 1, size=(N, 2))
A_labels = np.random.randint(0, 3, size=(N,))

# Ground Truth Ranks (1 is highest/best, N is lowest/worst)
truth_ranks = np.random.permutation(np.arange(1, N + 1))

# Create pairs (winner_idx, loser_idx) 
pairs = []
for i in range(N):
    for j in range(N):
        if truth_ranks[i] < truth_ranks[j]:
            pairs.append((i, j))
pairs = jnp.array(pairs)


# ==========================================
# 2. MODEL & LOSS DEFINITION
# ==========================================
def predict_scores(params, D, A):
    """ R_j = (a11 * D_j1 + a22 * D_j2) * A_weight[A_i] """
    a1 = params['a11'] * D[:, 0] + params['a22'] * D[:, 1]
    return a1 * params['A_weights'][A]

def ranking_loss(params, D, A, pairs):
    scores = predict_scores(params, D, A)
    logits = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    loss = -jnp.mean(jax.nn.log_sigmoid(logits))
    l2 = 0.01 * (params['a11']**2 + params['a22']**2 + jnp.sum(params['A_weights']**2))
    return loss + l2


# ==========================================
# 3. TRAINING LOOP
# ==========================================
params = {
    'a11': jnp.array(1.0),
    'a22': jnp.array(1.0),
    'A_weights': jnp.ones(3)
}

optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, D, A, pairs):
    loss_value, grads = jax.value_and_grad(ranking_loss)(params, D, A, pairs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

print("--- TRAINING ---")
for epoch in range(100):
    params, opt_state, loss_value = step(params, opt_state, D_features, A_labels, pairs)
    if epoch % 20 == 0 or epoch == 99:
        print(f"Epoch {epoch:02d} | Loss: {loss_value:.4f}")


# ==========================================
# 4. THE AUDIT: FRUSTRATION SCORES
# ==========================================
def calculate_frustration(params, D, A, pairs_array, N_scenarios):
    """
    Calculates how often the model contradicts the ground truth, 
    and groups those contradictions by scenario.
    """
    # Get final predicted scores
    final_scores = predict_scores(params, D, A)
    
    # Calculate probabilities for all truth pairs: P(Winner > Loser)
    logits = final_scores[pairs_array[:, 0]] - final_scores[pairs_array[:, 1]]
    probabilities = jax.nn.sigmoid(logits)
    
    # Identify Hard Violations (Model predicted Loser > Winner)
    hard_violations = probabilities < 0.5
    
    # Initialize tracking arrays for each scenario
    scenario_frustration = np.zeros(N_scenarios)
    scenario_involvement = np.zeros(N_scenarios)
    
    pairs_np = np.array(pairs_array)
    probs_np = np.array(probabilities)
    violations_np = np.array(hard_violations)
    
    # Accumulate frustration for every scenario involved in a pair
    for idx, (winner, loser) in enumerate(pairs_np):
        # Frustration is the remaining probability mass (1 - P)
        # If P is 0.99, frustration is 0.01. If P is 0.1, frustration is 0.9.
        frust_val = 1.0 - probs_np[idx]
        
        # Add to both winner and loser (both are part of a confusing pair)
        scenario_frustration[winner] += frust_val
        scenario_frustration[loser] += frust_val
        scenario_involvement[winner] += 1
        scenario_involvement[loser] += 1

    # Normalize by how many pairs the scenario was involved in
    avg_frustration = scenario_frustration / np.maximum(scenario_involvement, 1)
    
    print("\n--- FRUSTRATION REPORT (TOP 3 WORST OFFENDERS) ---")
    # Sort scenarios by highest average frustration
    worst_scenarios = np.argsort(avg_frustration)[::-1]
    
    for i in range(3):
        s_idx = worst_scenarios[i]
        print(f"Scenario ID: {s_idx:02d} | True Rank: {truth_ranks[s_idx]:02d} | Avg Frustration: {avg_frustration[s_idx]:.3f} | Labels: A_i={A_labels[s_idx]}")
        
    return final_scores, avg_frustration

final_scores, frustration_metrics = calculate_frustration(params, D_features, A_labels, pairs, N)