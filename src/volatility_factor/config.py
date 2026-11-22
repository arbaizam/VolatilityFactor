"""Model configuration (the ``var_config_volfactor.yml`` file).

Nothing is read at import time. The active configuration is resolved lazily, in order:

1. a config passed explicitly (``config=`` arguments, :func:`set_config`, :func:`use_config`)
2. the ``VOLFACTOR_CONFIG`` environment variable
3. ``configs/var_config_volfactor.yml`` in the working directory or any parent directory

Relative paths inside the file (``pricing_data``, ``loc_output``) are resolved against
the directory that contains the config file, so notebooks work from any folder.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "VOLFACTOR_CONFIG"
DEFAULT_CONFIG_RELPATH = Path("configs") / "var_config_volfactor.yml"


def _normalize_weights(value: Any) -> tuple[float, ...] | None:
    """YAML weights are either a list or the literal ``None`` / ``"None"``."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    return tuple(value)


@dataclass(frozen=True)
class FactorConfig:
    """One entry of ``var_settings``."""

    volatility_factor_id: str
    tickers: tuple[str, ...]
    ndays: tuple[int, ...]
    weights: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, volatility_factor_id: str, values: Mapping[str, Any]) -> FactorConfig:
        """Build from one ``var_settings`` entry.

        Args:
            volatility_factor_id: the entry's key.
            values: the entry (``tickers``, ``ndays`` and optional ``weights``).
        """
        return cls(
            volatility_factor_id=str(volatility_factor_id),
            tickers=tuple(values["tickers"]),
            ndays=tuple(int(n) for n in values["ndays"]),
            weights=_normalize_weights(values.get("weights")),
        )

    @property
    def ticker_list(self) -> list[str]:
        """Tickers as a list."""
        return list(self.tickers)

    @property
    def weight_list(self) -> list[float] | None:
        """Weights as a list, or ``None`` for equal weights."""
        return None if self.weights is None else list(self.weights)


@dataclass(frozen=True)
class GarchSettings:
    """The ``garch_settings`` section."""

    last_sample_date: str
    horizon1: int
    horizon2: int
    nsims: int
    number_of_periods: int
    sample_years: int
    forecast_months: int
    forecast_lag: int
    var_type: tuple[str, ...] = ("long_term",)
    start_point_quantiles: tuple[float, ...] = (0.50,)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GarchSettings:
        """Build from the ``garch_settings`` section.

        Args:
            values: the section as a mapping.
        """
        return cls(
            last_sample_date=str(values["last_sample_date"]),
            horizon1=int(values["horizon1"]),
            horizon2=int(values["horizon2"]),
            nsims=int(values["nsims"]),
            number_of_periods=int(values["number_of_periods"]),
            sample_years=int(values["sample_years"]),
            forecast_months=int(values["forecast_months"]),
            forecast_lag=int(values["forecast_lag"]),
            var_type=tuple(values.get("var_type", ("long_term",))),
            start_point_quantiles=tuple(values.get("start_point_quantiles", (0.50,))),
        )


@dataclass(frozen=True)
class Config:
    """Parsed configuration file.

    Example::

        cfg = Config.load("configs/var_config_volfactor.yml").with_garch(nsims=2_000)
        cfg.factor("agy_cmo").tickers
    """

    raw: Mapping[str, Any]
    path: Path | None = None
    _garch_overrides: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        """Load a YAML config. With no path, use the default resolution order."""
        resolved = Path(path) if path is not None else _discover_config_path()
        if resolved is None:
            raise FileNotFoundError(
                f"No config file found. Pass a path, set ${CONFIG_ENV_VAR}, or run from a "
                f"directory that contains {DEFAULT_CONFIG_RELPATH}."
            )
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(raw=raw, path=resolved.resolve())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base_dir: str | os.PathLike | None = None) -> Config:
        """Build a config from a dict. ``base_dir`` anchors relative paths."""
        path = Path(base_dir).resolve() / "<dict>" if base_dir is not None else None
        return cls(raw=dict(raw), path=path)

    # ----------------------------------------------------------------- chaining
    def with_garch(self, **overrides: Any) -> Config:
        """Return a copy with ``garch_settings`` values overridden (e.g. ``nsims=2000``)."""
        return replace(self, _garch_overrides={**self._garch_overrides, **overrides})

    def with_factors(self, *volatility_factor_ids: str) -> Config:
        """Return a copy restricted to the given factors (in the given order)."""
        settings = self.raw.get("var_settings", {})
        missing = [vf for vf in volatility_factor_ids if vf not in settings]
        if missing:
            raise KeyError(f"Unknown volatility_factor_id(s): {missing}")
        raw = {**self.raw, "var_settings": {vf: settings[vf] for vf in volatility_factor_ids}}
        return replace(self, raw=raw)

    # ----------------------------------------------------------------- sections
    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the config are resolved against."""
        return self.path.parent if self.path is not None else Path.cwd()

    @property
    def data_settings(self) -> Mapping[str, Any]:
        """The ``data_settings`` section (empty if absent)."""
        return self.raw.get("data_settings", {}) or {}

    @property
    def garch(self) -> GarchSettings:
        """The ``garch_settings`` section, with any ``with_garch`` overrides applied."""
        return GarchSettings.from_dict({**self.raw.get("garch_settings", {}), **self._garch_overrides})

    @property
    def factors(self) -> dict[str, FactorConfig]:
        """Configured factors keyed by ``volatility_factor_id``, in file order."""
        return {
            vf_id: FactorConfig.from_dict(vf_id, values)
            for vf_id, values in (self.raw.get("var_settings", {}) or {}).items()
        }

    def factor(self, volatility_factor_id: str) -> FactorConfig:
        """One configured factor.

        Args:
            volatility_factor_id: key in ``var_settings``.

        Raises:
            KeyError: if the id is not configured (the message lists the known ids).
        """
        try:
            return self.factors[volatility_factor_id]
        except KeyError:
            known = ", ".join(self.factors) or "<none>"
            raise KeyError(
                f"Unknown volatility_factor_id {volatility_factor_id!r}. Known ids: {known}"
            ) from None

    def find_factor_id(self, tickers: Sequence[str]) -> str | None:
        """The factor id configured for exactly these tickers (order matters; the last match wins)."""
        found = None
        for vf_id, factor in self.factors.items():
            if list(factor.tickers) == list(tickers):
                found = vf_id
        return found

    def resolve_path(self, value: str | os.PathLike) -> Path:
        """Resolve a path from the config file.

        Args:
            value: absolute path, or a path relative to the config file's directory.

        Returns:
            Absolute path.
        """
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def pricing_data_path(self) -> Path | None:
        """``data_settings.pricing_data`` as an absolute path, or ``None``."""
        value = self.data_settings.get("pricing_data")
        return self.resolve_path(value) if value else None

    @property
    def output_dir(self) -> Path:
        """``data_settings.loc_output`` as an absolute path (default ``output``)."""
        return self.resolve_path(self.data_settings.get("loc_output", "output"))

    @property
    def output_file(self) -> Path:
        """``loc_output / output_filename`` (the stem used for all saved outputs)."""
        return self.output_dir / self.data_settings.get("output_filename", "combined_vars.xlsx")

    def periods(self) -> tuple[list[list[str]], list[list[str]]]:
        """Sample and forecast windows from ``garch_settings`` (see :func:`create_periods`)."""
        from volatility_factor.periods import create_periods

        g = self.garch
        return create_periods(
            last_sample_date=g.last_sample_date,
            periods=g.number_of_periods,
            forecast_mos=g.forecast_months,
            forecast_lag=g.forecast_lag,
            sample_yrs=g.sample_years,
        )


# --------------------------------------------------------------------------- #
# Active (default) configuration
# --------------------------------------------------------------------------- #
_active_config: Config | None = None


def _discover_config_path() -> Path | None:
    """Find the default config file.

    Checks ``$VOLFACTOR_CONFIG``, then ``configs/var_config_volfactor.yml`` in the working
    directory and each parent directory.

    Returns:
        The path, or ``None`` if none is found.
    """
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env)
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / DEFAULT_CONFIG_RELPATH
        if candidate.is_file():
            return candidate
    return None


def _coerce_config(config: Config | str | os.PathLike | Mapping[str, Any]) -> Config:
    """Turn a :class:`Config`, path or dict into a :class:`Config`."""
    if isinstance(config, Config):
        return config
    if isinstance(config, Mapping):
        return Config.from_dict(config)
    return Config.load(config)


def get_config(config: Config | str | os.PathLike | Mapping[str, Any] | None = None) -> Config:
    """Return ``config`` coerced to :class:`Config`, or the active default config."""
    global _active_config
    if config is not None:
        return _coerce_config(config)
    if _active_config is None:
        _active_config = Config.load()
    return _active_config


def set_config(config: Config | str | os.PathLike | Mapping[str, Any] | None) -> Config | None:
    """Set the active default config (``None`` resets to automatic discovery)."""
    global _active_config
    _active_config = None if config is None else _coerce_config(config)
    return _active_config


@contextmanager
def use_config(config: Config | str | os.PathLike | Mapping[str, Any]) -> Iterator[Config]:
    """Temporarily make ``config`` the active default (handy in tests)."""
    global _active_config
    previous = _active_config
    _active_config = _coerce_config(config)
    try:
        yield _active_config
    finally:
        _active_config = previous
