import torch
import torch.nn as nn
from emlp.reps import T
import emlp.nn.pytorch as em
from emlp.groups import SO
from emlp.reps import T
import torch.nn.functional as F
import torch
import torch.nn as nn

class RotationEqActor(nn.Module):
    r"""
    Rotationally equivariant actor network that outputs a 2-D action vector (fx, fy),
    transforming under the standard SO(2) vector representation.

    This module uses an E(2)-equivariant MLP (EMLP) to guarantee that the output action
    transforms *equivariantly* with respect to rotations applied to the input state.
    In other words, if the entire spatial state (positions, velocities, etc.) is rotated
    by some θ ∈ SO(2), the output action will also rotate by the same θ.

    **Where and how the equivariance happens**
    -----------------------------------------
    Equivariance is enforced *entirely inside the EMLP architecture* from the `emlp`
    library:

    * The input representation `rep_in` is a direct sum of SO(2) vector reps (for
      2-D geometric quantities) and scalar reps (for rotation-invariant quantities).
    * The output representation `rep_out` is an SO(2) vector rep.
    * The `EMLP` constructor uses these representations + the `SO(2)` group to build:
        - **Equivariant linear layers**, whose weight matrices are constrained to
          commute with the group action.
        - **Equivariant nonlinearities**, guaranteeing that each layer preserves
          equivariance.
    * As a result, the entire mapping `x → emlp(x)` is equivariant under SO(2).

    Parameters
    ----------
    state_size : int
        Dimensionality of the input state (default: 18).
    action_size : int
        Dimensionality of the output action (default: 2).
    hidden_sizes : tuple
        Hidden channel sizes passed to the EMLP (the first is used as `ch`).
    init_w : float
        Unused here, but often used for output-layer init range.
    action_low : float
        Minimum value of each action dimension.
    action_high : float
        Maximum value of each action dimension.
    num_layers : int
        Number of equivariant layers in the EMLP.

    Notes
    -----
    The actor assumes a state layout of:
        - self velocity: 2D vector
        - self position: 2D vector
        - 3× landmarks: 2D vectors each → 6 dims
        - 2× other agents: 2D vectors each → 4 dims
        - 4× communication scalars: 4 dims - meh

    These features are concatenated in an order that matches the declared
    representation structure in `rep_in`.
    """

    def __init__(self, state_size=18, action_size=2,
                 hidden_sizes=(128,), init_w=3e-3,
                 action_low=-1, action_high=1,
                 num_layers=2):
        super().__init__()

        # Action scaling parameters
        self.scale = (action_high - action_low) / 2
        self.bias  = (action_high + action_low) / 2
        self.action_size = action_size

        # --- Define input and output representation types ---
        # T(1, 0) is the 2D vector representation of SO(2)
        # T(0, 0) is the scalar representation (invariant under rotation)
        vec = T(1, 0)
        scalar = T(0, 0)

        # Input representation = direct sum of reps in correct order
        self.rep_in = (
            vec +          # self_vel (2D)
            vec +          # self_pos (2D)
            3 * vec +      # 3 landmark vectors (each 2D)
            2 * vec +      # 2 other-agent vectors (each 2D)
            4 * scalar     # 4 scalar communication channels
        )

        # Output representation: a single SO(2) vector for (fx, fy)
        self.rep_out = vec

        # The symmetry group: planar rotations
        self.group = SO(2)

        # The equivariant network. All equivariance is enforced here through:
        #  - Representation-aware linear maps
        #  - Group-constrained parameter sharing
        #  - Equivariant nonlinearities
        self.emlp = em.EMLP(
            rep_in=self.rep_in,
            rep_out=self.rep_out,
            group=self.group,
            ch=128,
            num_layers=num_layers
        )
        
        # If action_size > 2, we need to add communication actions
        # Communication actions are scalars (not rotationally equivariant)
        if action_size > 2:
            comm_size = action_size - 2
            self.comm_head = nn.Sequential(
                nn.Linear(state_size, hidden_sizes[0] if isinstance(hidden_sizes, tuple) else hidden_sizes),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0] if isinstance(hidden_sizes, tuple) else hidden_sizes, comm_size),
                nn.Tanh()
            )
        else:
            self.comm_head = None

    def forward(self, state):
        r"""
        Forward pass through the actor.

        Parameters
        ----------
        state : torch.Tensor
            Shape (batch, 18) containing positions, velocities, landmarks,
            other agents, and scalar comm features.

        Returns
        -------
        action : torch.Tensor
            Shape (batch, 2). A rotationally equivariant 2-D action vector.

            The equivariance holds because the *internal mapping* `emlp(x)` is
            SO(2)-equivariant. The final tanh + affine scaling does not break
            equivariance because it acts component-wise and preserves the
            vector transformation properties.
        """

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # --- Extract and label each block of the state ---
        v_self = state[:, 0:2]     # SO(2) vector
        p_self = state[:, 2:4]     # SO(2) vector
        lmk    = state[:, 4:10]    # 3 × 2D landmark vectors
        other  = state[:, 10:14]   # 2 × 2D other-agent vectors
        comm   = state[:, 14:18]   # 4 scalar features

        # Split landmarks
        l1 = lmk[:, 0:2]
        l2 = lmk[:, 2:4]
        l3 = lmk[:, 4:6]

        # Split other agents
        o1 = other[:, 0:2]
        o2 = other[:, 2:4]

        # Build input in the *exact representation order* required by `rep_in`.
        # Order matters because EMLP uses representation structure to enforce symmetry.
        x = torch.cat([
            v_self, p_self,
            l1, l2, l3,
            o1, o2,
            comm[:, 0:1], comm[:, 1:2], comm[:, 2:3], comm[:, 3:4],
        ], dim=1)

        # Pass through equivariant MLP for 2D force vector
        force_action = torch.tanh(self.emlp(x))
        
        # Scale force to action range
        force_action = self.scale * force_action + self.bias
        
        # If we need communication actions, generate them separately
        if self.comm_head is not None:
            comm_action = self.comm_head(state)
            # Scale communication actions to [0, 1] range (typical for communication)
            comm_action = (comm_action + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            # Concatenate force and communication actions
            action = torch.cat([force_action, comm_action], dim=-1)
        else:
            action = force_action

        # Remove batch dimension if originally absent
        if action.shape[0] == 1 and state.dim() == 2:
            action = action.squeeze(0)

        return action

class GraphConvLayer(nn.Module):
    r"""
    A basic Graph Convolutional Network (GCN) layer.

    This implements the operation:
        H' = A · (W H)

    where:
        - H  : input feature matrix  [n_agents, input_dim]
        - W  : learnable weight matrix
        - A  : adjacency matrix (symmetric, normalized)
        - H' : output feature matrix [n_agents, output_dim]

    Note
    ----
    This layer **does not perform any normalization internally**;
    it assumes the caller provides an appropriately normalized
    adjacency matrix `input_adj`.
    """
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.lin_layer = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input_feature, input_adj):
        """
        Parameters
        ----------
        input_feature : torch.Tensor
            Shape [batch_size, n_agents, input_dim]
        input_adj : torch.Tensor
            Shape [n_agents, n_agents]. Shared across the batch.

        Returns
        -------
        torch.Tensor
            Shape [batch_size, n_agents, output_dim]
            Result of A·(W H)
        """
        # First map node features with a learnable linear transform.
        feat = self.lin_layer(input_feature)

        # Message passing: multiply adjacency with transformed features.
        out = torch.matmul(input_adj, feat)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})"



class GraphNet(nn.Module):
    r"""
    Graph neural network used to preprocess per-agent **state+action** inputs
    and achieve **permutation invariance w.r.t. agent ordering**.

    This solves the "order issue" that arises in multi-agent critics:
    the final value estimate should *not* depend on how agents are ordered
    in the input tensor.

    Architecture
    ------------
    - Two GCN layers: (gc1, gc2)
    - Two per-node MLP paths (nn_gc1, nn_gc2) acting as skip connections
    - Shared adjacency matrix where every agent is fully connected:
        A[i, j] = 1/n_agents  for i ≠ j
        A[i, i] = 0
    - Mean or max pooling across agents
    - Final scalar output V(s, a)

    Permutation Invariance
    ----------------------
    Because:
        1. All agents share the same adjacency structure,
        2. GCN layers treat all nodes symmetrically, and
        3. A symmetric pooling operator (avg or max) is applied,
    the resulting network is invariant to permutations of agents.
    """
    def __init__(self, sa_dim, n_agents, hidden_size, pool_type='avg'):
        super(GraphNet, self).__init__()

        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
      
        # Two GCN layers
        self.gc1 = GraphConvLayer(sa_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)

        # Skip/parallel MLP paths (node-wise)
        self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        # Final value head
        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Fully connected normalized adjacency (no self-loops).
        # This builds a permutation-invariant graph structure.
        self.register_buffer(
            'adj',
            (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / n_agents
        )

    def forward(self, x):
        r"""
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape [batch_size, n_agents, sa_dim]
            Each agent's concatenated (state, action) vector.

        Returns
        -------
        torch.Tensor
            Shape [batch_size, 1], value estimate V.
        """

        # --- First graph conv block with residual ---
        feat = F.relu(self.gc1(x, self.adj))          # Message passing
        feat = feat + F.relu(self.nn_gc1(x))          # Node-wise skip path
        feat = feat / float(self.n_agents)            # Normalize

        # --- Second graph conv block with residual ---
        out = F.relu(self.gc2(feat, self.adj))
        out = out + F.relu(self.nn_gc2(feat))
        out = out / float(self.n_agents)

        # --- Pool across agents (permutation-invariant) ---
        if self.pool_type == 'avg':
            ret = out.mean(1)
        elif self.pool_type == 'max':
            ret, _ = out.max(1)
        else:
            raise ValueError(f"Unknown pool_type {self.pool_type}")

        # Final scalar value head
        V = self.V(ret)
        return V



class PermutationInvCritic(nn.Module):
    r"""
    Permutation-invariant centralized critic for multi-agent RL.

    This critic:
        - Accepts the full joint state and joint action across all agents.
        - Splits them into per-agent (state, action) vectors.
        - Processes them with a permutation-invariant `GraphNet`.
        - Outputs a scalar joint value estimate V(s, a).

    This is commonly used in multi-agent actor–critic methods such as:
        - MADDPG-style critics
        - Centralized training, decentralized execution (CTDE)
    """
    def __init__(self, state_size, action_size, hidden_size, num_agents):
        super(PermutationInvCritic, self).__init__()

        self.num_agents = num_agents

        # Per-agent state+action dimension
        sa_dim = int((state_size + action_size) / num_agents)

        # GraphNet enforces permutation invariance over agents
        self.net = GraphNet(sa_dim, num_agents, hidden_size[0])
        
    def forward(self, inputs, actions):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape [batch_size, state_size]
        actions : torch.Tensor
            Shape [batch_size, action_size]

        Returns
        -------
        torch.Tensor
            Shape [batch_size, 1]
            Centralized value estimate V(s, a)
        """

        bz = inputs.size(0)

        # Reshape into [batch, agents, per-agent-state] and same for actions.
        s_n = inputs.view(bz, self.num_agents, -1)
        a_n = actions.view(bz, self.num_agents, -1)

        # Concatenate per-agent (state, action)
        x = torch.cat((s_n, a_n), dim=2)

        # Feed through permutation-invariant GNN
        V = self.net(x)
        return V

