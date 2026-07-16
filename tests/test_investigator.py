from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question


def investigate(investigator: DeploymentFailureInvestigator, question: str):
    return investigator.investigate(request_from_question(question))


def test_successful_root_cause_detection(investigator: DeploymentFailureInvestigator) -> None:
    result = investigate(investigator, "Why did deployment deploy-1042 fail?")

    assert result.root_cause == "The deployment health check timed out because the target service was unhealthy."
    assert result.confidence == 1.0
    assert result.limitations == ()


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
