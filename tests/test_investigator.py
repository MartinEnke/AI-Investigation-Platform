from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.models import Evidence


def investigate(investigator: DeploymentFailureInvestigator, question: str):
    return investigator.investigate(request_from_question(question))


def test_successful_root_cause_detection(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did deployment deploy-1042 fail?")

    assert result.root_cause == "The deployment health check timed out because the target service was unhealthy."
    assert result.answer == (
        "Deployment deploy-1042 failed during its health check. "
        "The deployment health check timed out because the target service was unhealthy."
    )
    assert result.confidence == 1.0
    assert result.limitations == ()
    assert result.evidence == (
        Evidence(
            source="deployment",
            summary="deploy-1042 has status failed and failed stage health_check.",
        ),
        Evidence(source="logs", summary="Health check timed out after 30 seconds."),
        Evidence(
            source="service_health",
            summary="checkout-api was unhealthy: Readiness endpoint returned HTTP 503.",
        ),
    )


def test_evidence_ordering(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did deployment deploy-1042 fail?")

    assert [item.source for item in result.evidence] == ["deployment", "logs", "service_health"]


def test_unknown_deployment(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did deploy-9999 fail?")

    assert result.root_cause is None
    assert result.evidence == ()
    assert result.confidence == 0.0
    assert "not found" in result.answer


def test_missing_deployment_id(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did the deployment fail?")

    assert result.root_cause is None
    assert result.evidence == ()
    assert result.confidence == 0.0
    assert "no deployment ID" in result.answer


def test_inconclusive_result_with_incomplete_evidence(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did deployment deploy-1043 fail?")

    assert result.root_cause is None
    assert [item.source for item in result.evidence] == ["deployment"]
    assert result.confidence == 0.25
    assert result.answer.endswith("is inconclusive.")
    assert result.limitations == (
        "No error log is available for the deployment.",
        "No service-health record is available for the target service.",
    )


def test_multiple_matches_cause_explicit_abstention() -> None:
    class Deployments:
        def get_deployment(self, deployment_id: str):
            return {"service": "api", "status": "failed", "failed_stage": "startup"}

    class Logs:
        def get_logs(self, deployment_id: str):
            return [
                {"level": "error", "message": "Missing environment variable DATABASE_URL"},
                {"level": "error", "message": "Migration failed"},
            ]

    class Health:
        def get_service_health(self, service: str):
            return None

    result = investigate(
        DeploymentFailureInvestigator(Deployments(), Logs(), Health()),
        "Why did deployment deploy-2000 fail?",
    )

    assert result.root_cause is None
    assert result.confidence == 0.25
    assert result.limitations == (
        "Conflicting supported failure patterns matched the available evidence.",
    )
    assert [item.summary for item in result.evidence[1:]] == [
        "Missing environment variable DATABASE_URL",
        "Migration failed",
    ]


def test_non_error_diagnostic_log_does_not_trigger_diagnosis() -> None:
    class Deployments:
        def get_deployment(self, deployment_id: str):
            return {"service": "api", "status": "failed", "failed_stage": "startup"}

    class Logs:
        def get_logs(self, deployment_id: str):
            return [
                {
                    "deployment_id": deployment_id,
                    "level": "warning",
                    "message": "Missing environment variable DATABASE_URL",
                }
            ]

    class Health:
        def get_service_health(self, service: str):
            return None

    result = investigate(
        DeploymentFailureInvestigator(Deployments(), Logs(), Health()),
        "Why did deployment deploy-2000 fail?",
    )

    assert result.root_cause is None
    assert result.evidence == (
        Evidence(
            source="deployment",
            summary="deploy-2000 has status failed and failed stage startup.",
        ),
    )
    assert result.limitations == (
        "No error log is available for the deployment.",
        "No service-health record is available for the target service.",
    )


def test_error_for_another_deployment_does_not_influence_request(
    investigator: DeploymentFailureInvestigator,
) -> None:
    result = investigate(investigator, "Why did deployment deploy-1046 fail?")

    assert result.root_cause is None
    assert [item.summary for item in result.evidence if item.source == "logs"] == [
        "Rollout was denied by the deployment policy."
    ]
    assert result.limitations == (
        "The available evidence does not match a known failure pattern.",
    )
