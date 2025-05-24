import torch
import torch.nn as nn
import numpy as np

     
class LISTA(nn.Module):
    """
        Learned ISTA (LISTA) Network.

        This model unrolls the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse signal recovery
        into a trainable neural network with `numLayers` layers. Each layer mimics an ISTA iteration and
        optionally learns its own parameters.

        Parameters:
        -----------
        numLayers : int
            Number of LISTA layers to unroll (equivalent to ISTA iterations).

        szA : tuple
            Shape of the dictionary matrix A as (rows, cols). Only used if A is not provided.

        scale_mag : float
            Initial scale factor used when initializing weights (or for computing preset W/S if A is provided).

        actfunc : str, optional (default="shrink")
            Activation function to use per layer.
            Options:
                - "shrink": soft thresholding (default, L1-based sparsity)
                - "swish": SiLU activation
                - "tanh" : Hyperbolic tangent

        A : np.ndarray or None, optional (default=None)
            Fixed dictionary matrix A (size: rows × cols). If None, A is initialized randomly.

        untied : bool, optional (default=False)
            If True, each LISTA layer will have its own learnable S matrix and λ parameter.
            If False, all layers share the same S and λ (tied weights).

        untrained_lamL1 : float or None, optional (default=None)
            If provided, sets λ to a fixed scalar value (not learned).
            If None, λ is learned during training.

        Attributes:
        -----------
        W : torch.nn.Parameter
            Weight matrix equivalent to γ·Aᵗ (shared or fixed).

        S : torch.nn.Parameter or list of Parameters
            Matrix controlling the recursive residual update per layer.
            If `untied`, this is a list of Parameters (one per layer).

        lam : torch.nn.Parameter or list of Parameters
            Shrinkage threshold(s). Can be fixed or learned.
   """

    def __init__(self, numLayers, szA, scale_mag, actfunc="shrink", untrained_lamL1=None, untied=False, A=None):
        super().__init__()
        print("Inside lista class")

        if A is None:
            self.szA_0, self.szA_1 = szA
            self.predefined_A = False
        else:
            self.szA_0, self.szA_1 = A.shape
            AT = A.T
            ATA = AT @ A
            stp = scale_mag
            self.preset_W = (1 / stp) * AT
            self.preset_S = np.eye(self.szA_1) - ((1 / stp) * (AT @ A))
            self.predefined_A = True

        self.NL = numLayers
        self.untied = untied
        self.actfunc = actfunc
        self.scale_mag = scale_mag
        self.untrained_lamL1 = untrained_lamL1

        print("Inside lista class 1")

        if (untied):
            print("Inside lista class 2")
            W, _, lam = self.initVars()
            self.W = W  # W is always tied
            self.S = []  # If 0 layers, do not need S

            self.register_parameter("lam0", lam)
            self.lam = [lam]
            print("Inside lista class 3")

            for L in range(self.NL):
                print(L)
                _, S, lam = self.initVars()
                print("Inside lista class 3-1")
                self.register_parameter("S%d" % L, S)
                print("Inside lista class 3-2")
                self.S.append(S)
                print("Inside lista class 3-3")

                self.register_parameter("lam%d" % (L + 1), lam)
                print("Inside lista class 3-4")
                self.lam.append(lam)
                print("Inside lista class 3-5")
            print("Inside lista class 4")

        else:
            print("Inside lista class 5")
            W, S, lam = self.initVars()
            self.W = W
            self.S = S
            self.lam = lam

    def initVars(self):
        print("Init vars")
        if self.predefined_A:
            print("Init vars 1")
            W_np = self.preset_W
            S_np = self.preset_S
        else:
            print("Init vars 2")
            # W_np = self.scale_mag * np.random.rand(self.szA_1, self.szA_0)
            # S_np = self.scale_mag * np.random.rand(self.szA_1, self.szA_1)
            W_torch = self.scale_mag * torch.rand((self.szA_1, self.szA_0), device='cuda')
            S_torch = self.scale_mag * torch.rand((self.szA_1, self.szA_1), device='cuda')

            # Optional: convert back to NumPy if needed
            W_np = W_torch.cpu().numpy()
            S_np = S_torch.cpu().numpy()
        print("Init vars 3")
        lam_np = self.scale_mag * np.random.rand(1, 1)

        print("Init vars 4")
        W = nn.Parameter(torch.tensor(W_np, dtype=torch.double))
        print("Init vars 5")
        S = nn.Parameter(torch.tensor(S_np, dtype=torch.double))
        print("Init vars 6")

        if (self.untrained_lamL1 == None):
            print("Init vars 7")
            lam = nn.Parameter(torch.tensor(lam_np, dtype=torch.double, requires_grad=True))
        else:
            print("Init vars 8")
            lam = torch.tensor(self.untrained_lamL1, dtype=torch.double, requires_grad=False)

        print("Init vars 9")
        return W, S, lam

    def applyActFunc(self, x, L=None):
        if self.actfunc == "shrink":
            if self.untied:
                return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam[L], min=0)
            else:
                return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam, min=0)
        elif self.actfunc == "swish":
            return torch.nn.SiLU()(x)
        elif self.actfunc == "tanh":
            return torch.nn.Tanh()(x)
        else:
            raise ValueError('Incorrect activation function set')

    def forward(self, Y):
        if (self.untied):
            Wy = self.W @ Y
            Z = Wy
            for L in range(self.NL):
                Z = Wy + self.S[L] @ self.applyActFunc(Z, L)
            return self.applyActFunc(Z, self.NL)

        else:
            Wy = self.W @ Y
            Z = Wy
            for L in range(self.NL):
                Z = Wy + self.S @ self.applyActFunc(Z)
            return self.applyActFunc(Z)