import xarray as xr
import assetra
from assetra.units import EnergyUnit
import logging 
import numpy as np

from dataclasses import dataclass
import xarray as xr
import numpy as np
from dataclasses import dataclass
import xarray as xr
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class ARSStorageUnit(EnergyUnit):
    """Derived energy unit class.

    A storage unit is a state-dependent, responsive energy unit. 
    The available capacity of a storage unit depends on its state 
    of charge and on the needs of the system. As opposed to static 
    and stochastic units, which require hourly time series, storage 
    unit operation is characterized by a handful of scalar parameters.

    Args:
        id (int): Unique identifying number
        nameplate_capacity (float): Nameplate capacity in units of power. 
            For storage, typically the discharge rate
        charge_rate (float): Charge rate in units of power
        discharge_rate (float): Discharge rate in units of power
        charge_capacity (float): Maximum charge capacity in units of energy
        roundtrip_efficiency (float): Roundtrip efficiency as decimal percent
        net_hourly_capacity (xr.DataArray, optional): Net capacity contained in
            DataArray with dimension (time) and hourly datetime coordinates
    """

    charge_rate: float
    discharge_rate: float
    charge_capacity: float
    roundtrip_efficiency: float
    net_hourly_capacity: xr.DataArray = None

    @staticmethod
    def _get_hourly_capacity(
        charge_rate: float,
        discharge_rate: float,
        charge_capacity: float,
        roundtrip_efficiency: float,
        net_hourly_capacity: xr.DataArray,
    ) -> xr.DataArray:
        """Greedy storage dispatch.

        Args:
            charge_rate (float): Charge rate in units of power.
            discharge_rate (float): Discharge rate in units of power.
            charge_capacity (float): Maximum charge capacity in units of energy.
            roundtrip_efficiency (float): Roundtrip efficiency as decimal percent.
            net_hourly_capacity (xr.DataArray): Net capacity contained
                in DataArray with dimension (time) and hourly datetime coordinates.

        Returns:
            xr.DataArray: Hourly capacity contained in DataArray with the same shape as net hourly capacity.
        """
        efficiency = roundtrip_efficiency ** 0.5

        def charge_storage(excess_capacity: float, current_charge: float):
            capacity = -min(
                charge_rate,
                (charge_capacity - current_charge) / efficiency,
                excess_capacity,
            )
            current_charge -= capacity * efficiency
            return capacity, current_charge

        def discharge_storage(unmet_demand: float, current_charge: float):
            capacity = min(
                discharge_rate / efficiency,
                current_charge,
                unmet_demand / efficiency,
            )
            current_charge -= capacity
            return capacity * efficiency, current_charge

        def dispatch_storage(net_hourly_capacity: xr.DataArray):
            current_charge = float(charge_capacity)  # Start with the max charge capacity
            for idx, net_capacity in enumerate(net_hourly_capacity):
                capacity = 0  # Assume no charge/discharge initially
                if net_capacity < 0:
                    if current_charge > 0:
                        capacity, current_charge = discharge_storage(
                            -net_capacity, current_charge
                        )
                elif current_charge < charge_capacity:
                    capacity, current_charge = charge_storage(
                        net_capacity, current_charge
                    )
                #LOG.debug(
                 #   f"Index: {idx}, net_capacity: {net_capacity}, capacity: {capacity}, current_charge: {current_charge}"
                #)
                yield capacity

        hourly_capacity = xr.DataArray(
            data=[
                capacity for capacity in dispatch_storage(net_hourly_capacity.values)
            ],
            dims=net_hourly_capacity.dims,
            coords=net_hourly_capacity.coords,
        )
        return hourly_capacity

    @staticmethod
    def to_unit_dataset(units: list['ARSStorageUnit']) -> xr.Dataset:
        """Convert a list of storage units into an xarray dataset.

        Args:
            units (list['ARSStorageUnit']): List of storage energy units.

        Returns:
            xr.Dataset: Dataset with dimensions (energy_unit) and variables 
            (nameplate_capacity[energy_unit], charge_rate[energy_unit], 
            discharge_rate[energy_unit], charge_capacity[energy_unit], 
            roundtrip_efficiency[energy_unit]).
            Coordinates for the energy_unit dimension are energy unit IDs.
        """
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(
                    ["energy_unit"],
                    [unit.nameplate_capacity for unit in units],
                ),
                charge_rate=(
                    ["energy_unit"],
                    [unit.charge_rate for unit in units],
                ),
                discharge_rate=(
                    ["energy_unit"],
                    [unit.discharge_rate for unit in units],
                ),
                charge_capacity=(
                    ["energy_unit"],
                    [unit.charge_capacity for unit in units],
                ),
                roundtrip_efficiency=(
                    ["energy_unit"],
                    [unit.roundtrip_efficiency for unit in units],
                ),
            ),
            coords=dict(energy_unit=[unit.id for unit in units]),
        )
        return unit_dataset

    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> list['ARSStorageUnit']:
        """Convert a storage unit dataset to a list of storage units.

        This is the inverse ARSStorageUnit.to_unit_dataset function.

        Args:
            unit_dataset (xr.Dataset): Unit dataset with structure and content defined 
            in the derived ARSStorageUnit.to_unit_dataset function.

        Returns:
            list['ARSStorageUnit']: List of storage units.
        """
        units = []
        for id, nc, cr, dr, cc, re in zip(
            unit_dataset.energy_unit,
            unit_dataset.nameplate_capacity,
            unit_dataset.charge_rate,
            unit_dataset.discharge_rate,
            unit_dataset.charge_capacity,
            unit_dataset.roundtrip_efficiency,
        ):
            units.append(
                ARSStorageUnit(
                    id=int(id),
                    nameplate_capacity=float(nc),
                    charge_rate=float(cr),
                    discharge_rate=float(dr),
                    charge_capacity=float(cc),
                    roundtrip_efficiency=float(re),
                )
            )
        return units

    @staticmethod
    def get_probabilistic_capacity_matrix(
        unit_dataset: xr.Dataset, net_hourly_capacity_matrix: xr.DataArray
    ) -> xr.DataArray:
        """Return probabilistic hourly capacity matrix for a storage unit dataset.

        For storage units, it is necessary to dispatch units every hour and iteration sequentially. 
        The dispatch policy implemented in ARSStorageUnit._get_hourly_capacity is a greedy 
        policy to minimize expected unserved energy. Units are dispatched according to the order 
        they appear in the unit dataset.

        Args:
            unit_dataset (xr.Dataset): Storage unit dataset, as generated by 
            ARSStorageUnit.to_unit_dataset function.
            net_hourly_capacity_matrix (xr.DataArray): Probabilistic net hourly capacity matrix 
            with dimensions (trial, time).

        Returns:
            xr.DataArray: Combined hourly capacity for all units in the unit dataset 
            with the same dimensions and shape as net hourly capacity matrix.
        """
        LOG.info(f"net_hourly_capacity_matrix dimensions: {net_hourly_capacity_matrix.dims}")
        units = ARSStorageUnit.from_unit_dataset(unit_dataset)
        net_adj_hourly_capacity_matrix = net_hourly_capacity_matrix.copy()

        for idx, unit in enumerate(units):
            LOG.info(
                "Dispatching storage unit "
                + str(idx)
                + " of "
                + str(len(units))
                + " in all trials"
            )
            
            for trial in net_adj_hourly_capacity_matrix:
                trial += StorageUnit._get_hourly_capacity(
                    unit.charge_rate,
                    unit.discharge_rate,
                    unit.charge_capacity,
                    unit.roundtrip_efficiency,
                    trial,
                )

        return net_adj_hourly_capacity_matrix - net_hourly_capacity_matrix
