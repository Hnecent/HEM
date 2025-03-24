from typing import Any, Mapping, Optional


class AirHeatDynamics:
    """
    building air heat dynamics
    -----------------------
    Attributes:
        A_fen: float: 该房间的窗户面积 m^2
        V: float: 该房间的体积 m^3
        A_suf: float: 该房间的外墙面积 m^2
        C_air: float: 该房间的空气比热容 Wh/kgK
        p_air: float: 该房间的空气密度 kg/m^3
        U: float: 该房间的外墙传热系数 W/m^2K
        ACH: float: 该房间的空气变化率 h^-1
        SHGC: float: 该房间的窗户的太阳热辐射增益系数
        E_air: float: 该房间上升一度所需能量 Wh/K E_air = C_air * p_air * V, C_air=0.3 Wh/kgK, p_air=1.3 kg/m^3, V=200 m^3
    """
    default = {
        'minutes_per_time_step': 10,
        'init_indoor_temperature': 26,
        'A_fen': 64,
        'V': 2400,
        'A_suf': 1480,
        'C_air': 0.3,
        'p_air': 1.3,
        'U': 0.53,
        'ACH': 0.4,
        'SHGC': 0.3
    }

    def __init__(self, minutes_per_time_step: int, A_fen: float, V: float, A_suf: float, C_air: float, p_air: float,
                 U: float, ACH: float, SHGC: float, **kwargs: Any):
        # 属性
        self.hours_per_time_step = minutes_per_time_step / 60
        self.A_fen = A_fen
        self.V = V
        self.A_suf = A_suf
        self.C_air = C_air
        self.p_air = p_air
        self.U = U
        self.ACH = ACH  # air change per hour
        self.SHGC = SHGC  # solar heat gain coefficient
        self.E_air = self.C_air * self.p_air * self.V

        # 变动参数
        self.indoor_temperature:Optional[float] = None

    def update(self, outdoor_temperature, solar_radiation, heat_pump_out_power: float):
        P_con = self.U * self.A_suf * (outdoor_temperature - self.indoor_temperature)
        P_ven = self.E_air * self.ACH * (
                outdoor_temperature - self.indoor_temperature)
        P_solar = solar_radiation * self.A_fen * self.SHGC

        Q_gain = (P_con + P_ven + P_solar + heat_pump_out_power * 1000) * self.hours_per_time_step
        self.indoor_temperature = self.indoor_temperature + (Q_gain / self.E_air)

    def reset(self, init_indoor_temperature: float):
        self.indoor_temperature = init_indoor_temperature

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            'hours_per_time_step': self.hours_per_time_step,
            'A_fen': self.A_fen,
            'V': self.V,
            'A_suf': self.A_suf,
            'C_air': self.C_air,
            'p_air': self.p_air,
            'U': self.U,
            'ACH': self.ACH,
            'SHGC': self.SHGC,
            'E_air': self.E_air,
        }
