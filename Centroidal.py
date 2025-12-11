import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CentroidalModelInfoSimple:
    """Minimal subset of centroidal model info used by the Python WBC demo.

    This dataclass now carries a device/dtype so stored state/input are
    converted to the desired device (GPU when available) automatically when
    `update_state_input` is called.
    """
    generalizedCoordinatesNum: int
    actuatedDofNum: int
    numThreeDofContacts: int
    robotMass: float = 1.0
    nv: int = field(init=False)
    nu: int = field(init=False)
    # optional device/dtype configuration for stored tensors
    device: Optional[torch.device] = field(default=None)
    dtype: torch.dtype = field(default=torch.float64)
    state: Optional[torch.Tensor] = field(default=None, init=False)
    input: Optional[torch.Tensor] = field(default=None, init=False)

    def __post_init__(self):
        # common convention: nv = actuatedDofNum + 7 (floating base), nu = actuatedDofNum + 6
        self.nv = self.actuatedDofNum + 7
        self.nu = self.actuatedDofNum + 6
        # decide default device: prefer CUDA if available
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ensure dtype is set
        if self.dtype is None:
            self.dtype = torch.float64
        self.state = None
        self.input = None

    def update_state_input(self, state: torch.Tensor, input: torch.Tensor):
        """Store state/input as torch tensors on the configured device/dtype.

        Accepts Python sequences or torch tensors. Conversion is performed with
        torch.as_tensor and `.to(device,dtype)` to ensure downstream calls
        receive GPU tensors when CUDA is available.
        """
        # convert to tensor then move to configured device/dtype
        s = torch.as_tensor(state, dtype=self.dtype)
        u = torch.as_tensor(input, dtype=self.dtype)
        # move to device
        self.state = s.to(device=self.device, dtype=self.dtype)
        self.input = u.to(device=self.device, dtype=self.dtype)

    def getInfo(self):
        return self

    def getnv(self):
        return self.nv

    def getnu(self):
        return self.nu

    def getstate(self):
        return self.state

    def getinput(self):
        return self.input

    def getmass(self):
        return self.robotMass

    def getnumThreeDofContacts(self):
        return self.numThreeDofContacts

    def getactuatedDofNum(self):
        return self.actuatedDofNum

    def getgeneralizedCoordinatesNum(self):
        return self.generalizedCoordinatesNum

