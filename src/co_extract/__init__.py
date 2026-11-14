"""Change-order extraction pipeline: messy change order (PDF or text) -> validated JSON."""

__version__ = "0.1.0"

FIELDS = (
    "project_name",
    "contract_number",
    "change_order_number",
    "date",
    "requested_by",
    "description",
    "amount",
    "schedule_impact_days",
    "approval_status",
)
