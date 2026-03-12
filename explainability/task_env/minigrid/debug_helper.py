import flax.nnx as nnx
import jax.numpy as jnp

def compare_q_states(q1, q2, pruned=False):
    """
    Compare the parameters of two models and print where they differ.

    Args:
        q1: First model (e.g., a Flax/JAX model)
        q2: Second model (same structure as q1)
    """
    q1_state = nnx.state(q1)
    q2_state = nnx.state(q2)

    for layer_name, q1_param in q1_state.items():

        q2_param = q2_state[layer_name]

        if layer_name == "hidden_layers":
            for idx, q1_layer_params in q1_param.items():
                q2_layer_params = q2_param[idx]

                for pname, q1_val in q1_layer_params.items():
                    q2_val = q2_layer_params[pname]

                    val1 = jnp.array(q1_val.value)
                    val2 = jnp.array(q2_val.value)

                    diff_mask = val1 != val2
                    if jnp.any(diff_mask):
                        diff_positions = jnp.argwhere(diff_mask)
                        for pos in diff_positions:
                            pos_tuple = tuple(pos.tolist())
                            if pruned and val2[pos_tuple]==0.0:
                                continue
                            print(f"{layer_name}_{idx}: {pname} differs at {pos_tuple}: q1={val1[pos_tuple]}, q2={val2[pos_tuple]}")

        elif layer_name == "output_layer":
            for pname, q1_val in q1_param.items():
                q2_val = q2_param[pname]

                val1 = jnp.array(q1_val.value)
                val2 = jnp.array(q2_val.value)

                diff_mask = val1 != val2
                if jnp.any(diff_mask):
                    diff_positions = jnp.argwhere(diff_mask)
                    for pos in diff_positions:
                        pos_tuple = tuple(pos.tolist())
                        if pruned and (val2[pos_tuple]==0.0):
                            continue
                        print(f"{layer_name}: {pname} differs at {pos_tuple}: q1={val1[pos_tuple]}, q2={val2[pos_tuple]}")

        else:
            raise ValueError(f"Unexpected layer: {layer_name}")


def compare_mask_states(m1, m2):
    s1_state = nnx.state(m1, nnx.Param)
    s2_state = nnx.state(m2, nnx.Param)

    q1_state = s1_state.mask_logits
    q2_state = s2_state.mask_logits

    """
    Compare the parameters of two models and print where they differ.

    Args:
        m1: First model (e.g., a Flax/JAX model)
        m2: Second model (same structure as m1)
    """

    for layer_name, q1_param in q1_state.items():

        q2_param = q2_state[layer_name]

        if layer_name == "hidden_layers":
            for idx, q1_layer_params in q1_param.items():
                q2_layer_params = q2_param[idx]

                for pname, q1_val in q1_layer_params.items():
                    q2_val = q2_layer_params[pname]

                    val1 = jnp.array(q1_val.value)
                    val2 = jnp.array(q2_val.value)

                    # # print difference
                    # diff_mask = val1 != val2
                    # if jnp.any(diff_mask):
                    #     diff_positions = jnp.argwhere(diff_mask)
                    #     for pos in diff_positions:
                    #         pos_tuple = tuple(pos.tolist())
                    #         print(f"{layer_name}_{idx}: {pname} differs at {pos_tuple}: m1={val1[pos_tuple]}, m2={val2[pos_tuple]}")
                    
                    # print same value
                    same_mask = val1 == val2
                    if jnp.any(same_mask):
                        same_positions = jnp.argwhere(same_mask)
                        for pos in same_positions:
                            pos_tuple = tuple(pos.tolist())
                            print(f"{layer_name}_{idx}: {pname} equals at {pos_tuple}: q1={val1[pos_tuple]}, q2={val2[pos_tuple]}")

        elif layer_name == "output_layer":
            for pname, q1_val in q1_param.items():
                q2_val = q2_param[pname]

                val1 = jnp.array(q1_val.value)
                val2 = jnp.array(q2_val.value)

                # diff_mask = val1 != val2
                # if jnp.any(diff_mask):
                #     diff_positions = jnp.argwhere(diff_mask)
                #     for pos in diff_positions:
                #         pos_tuple = tuple(pos.tolist())
                #         print(f"{layer_name}: {pname} differs at {pos_tuple}: m1={val1[pos_tuple]}, m2={val2[pos_tuple]}")
                
                same_mask = val1 == val2
                if jnp.any(same_mask):
                    same_positions = jnp.argwhere(same_mask)
                    for pos in same_positions:
                        pos_tuple = tuple(pos.tolist())
                        print(f"{layer_name}: {pname} equals at {pos_tuple}: q1={val1[pos_tuple]}, q2={val2[pos_tuple]}")

        else:
            raise ValueError(f"Unexpected layer: {layer_name}")