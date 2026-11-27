"""Phase 2 Monitoring Server: query one case's metrics, logs and traces through MCP.

The tools are methods of :class:`MonitoringService`, usable without MCP::

    from aiops.monitoring import MonitoringService
    service = MonitoringService()               # $AIOPS_DATA_DIR or ./data
    service.detect_anomalies("RE2-OB-0078b4ee")

``aiops-monitoring-server`` serves them over MCP (see :mod:`aiops.monitoring.server`).
"""

from aiops.monitoring.service import MonitoringService, normalize_case_id

__all__ = ["MonitoringService", "normalize_case_id"]
