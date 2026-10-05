"""Curated failure signatures: probable cause + remediation for well-known log patterns.

This is the deterministic fallback when no LLM is available, and it is also passed to the LLM as
a hint (small local models produce markedly better root-cause analyses with it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeRule:
    name: str
    title: str
    pattern: re.Pattern[str]
    cause: str
    remediation: tuple[str, ...]
    owner: str


_TITLES = {
    "aws-ssm-credentials": "Restore SSM Agent IAM credentials",
    "smtp-port-25-blocked": "Fix outbound SMTP delivery",
    "oom-kill": "Mitigate memory exhaustion",
    "disk-full": "Free disk space / repair filesystem",
    "dns-failure": "Fix DNS resolution",
    "tls-failure": "Fix TLS certificates",
    "database-pressure": "Relieve database pressure",
    "ssh-bruteforce": "Harden SSH against brute force",
    "jwt-auth-failure": "Fix client token handling",
    "django-anonymous-fk": "Guard audit logging against anonymous users",
    "authorization-denied": "Review authorization rules",
    "segfault": "Investigate native crash",
    "systemd-unit-failure": "Repair failing systemd unit",
    "python-runtime-error": "Fix application exception",
    "upstream-unavailable": "Restore upstream dependency",
    "process-restart": "Investigate process restarts",
    "django-field-error": "Fix invalid ORM field lookups",
    "log-flood": "Reduce application log noise",
}


def _rule(name: str, pattern: str, cause: str, remediation: tuple[str, ...], owner: str) -> KnowledgeRule:
    return KnowledgeRule(name, _TITLES[name], re.compile(pattern, re.IGNORECASE), cause, remediation, owner)


DEFAULT_RULES: tuple[KnowledgeRule, ...] = (
    _rule(
        "aws-ssm-credentials",
        r"EC2RoleProvider|EC2Metadata|SSM role|instance profile role|RequestManagedInstanceRoleToken",
        "The AWS SSM Agent cannot obtain IAM credentials: no instance profile is attached (or it lacks SSM "
        "permissions), Default Host Management Configuration is not enabled (AccessDeniedException: instance "
        "management role is not configured), IMDS is unreachable, or the IMDSv2 hop limit is too low.",
        (
            "Confirm an IAM instance profile is attached and includes the AmazonSSMManagedInstanceCore policy, or "
            "enable Default Host Management Configuration in Systems Manager Fleet Manager.",
            "Check IMDS from the host: TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token "
            "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60') && curl -sH \"X-aws-ec2-metadata-token: $TOKEN\" "
            "http://169.254.169.254/latest/meta-data/iam/info",
            "If calls traverse a container or proxy, raise the IMDSv2 hop limit: aws ec2 "
            "modify-instance-metadata-options --instance-id <id> --http-put-response-hop-limit 2",
            "Restart the agent once fixed: sudo snap restart amazon-ssm-agent (or systemctl restart amazon-ssm-agent).",
        ),
        "cloud platform",
    ),
    _rule(
        "smtp-port-25-blocked",
        r"connect to \S+:25: (?:connection timed out|network is unreachable|connection refused)"
        r"|(postfix|smtp).*(:25\b|port 25).*(timed out|refused|unreachable)",
        "Outbound SMTP on port 25 is blocked or throttled (AWS blocks port 25 from EC2 by default; 'Network is "
        "unreachable' on IPv6 means the host has no IPv6 route), so mail delivery is deferred and notifications "
        "are delayed or lost.",
        (
            "Relay through an authenticated submission service on port 587 (Amazon SES, Microsoft 365 SMTP AUTH) "
            "by setting relayhost and SASL credentials in postfix main.cf.",
            "Alternatively request removal of the EC2 port-25 restriction from AWS.",
            "Inspect and flush the queue after the fix: postqueue -p; postqueue -f.",
        ),
        "platform / mail",
    ),
    _rule(
        "oom-kill",
        r"out of memory|oom-kill|invoked oom-killer|cannot allocate memory",
        "Memory exhaustion: the kernel OOM killer terminated a process.",
        (
            "Identify the victim and its RSS: dmesg -T | grep -i -A20 'out of memory'.",
            "Set memory limits and fix leaks; add swap or right-size the instance.",
            "Alert on memory saturation (>90%) before the OOM killer fires.",
        ),
        "platform",
    ),
    _rule(
        "disk-full",
        r"no space left on device|disk full|read-only file system",
        "Disk space is exhausted, or the filesystem was remounted read-only after I/O errors.",
        (
            "Find large paths: sudo du -xh / --max-depth=2 | sort -h | tail -20.",
            "Rotate and compress logs (logrotate), prune old artifacts and container images.",
            "If read-only, check dmesg for I/O errors before remounting.",
        ),
        "platform",
    ),
    _rule(
        "dns-failure",
        r"temporary failure in name resolution|name or service not known|could not resolve|NXDOMAIN|SERVFAIL",
        "DNS resolution is failing for a dependency.",
        (
            "Test resolution from the host: resolvectl query <name> / dig <name>.",
            "Check /etc/resolv.conf, systemd-resolved status and VPC DNS settings.",
            "Cache DNS results in clients with a sane TTL and retry on SERVFAIL.",
        ),
        "platform / network",
    ),
    _rule(
        "tls-failure",
        r"certificate (?:has )?expired|certificate verify failed|ssl handshake|tls handshake|x509",
        "TLS handshakes fail: expired or untrusted certificate, or protocol/cipher mismatch.",
        (
            "Inspect the certificate chain: openssl s_client -connect <host>:443 -servername <host>.",
            "Renew certificates and automate renewal (certbot/ACM); refresh the CA bundle on clients.",
        ),
        "platform / security",
    ),
    _rule(
        "database-pressure",
        r"too many connections|deadlock|connection pool|could not connect to server|OperationalError"
        r"|lock wait timeout",
        "The database is saturated or unreachable (connection exhaustion, lock contention or outage).",
        (
            "Check active connections and locks (pg_stat_activity / SHOW PROCESSLIST).",
            "Introduce or tune a connection pool (pgbouncer) and cap per-worker connections.",
            "Find and fix long transactions or missing indexes behind lock waits.",
        ),
        "database",
    ),
    _rule(
        "ssh-bruteforce",
        r"failed password|invalid user|maximum authentication attempts|authentication failure.*sshd",
        "Repeated SSH authentication failures: likely automated brute-force attempts.",
        (
            "Disable password authentication (PasswordAuthentication no) and use keys or SSM Session Manager.",
            "Enable fail2ban or restrict port 22 to known CIDRs via the security group.",
        ),
        "security",
    ),
    _rule(
        "django-anonymous-fk",
        r"Cannot assign .*AnonymousUser.*must be a .* instance",
        "Request/audit logging code assigns request.user to a foreign key while the request is unauthenticated "
        "(AnonymousUser), so Django raises ValueError and the log record is lost. It fires on every anonymous "
        "or expired-token request.",
        (
            "Guard the assignment: created_by = request.user if request.user.is_authenticated else None "
            "(make the FK null=True, or skip logging anonymous requests).",
            "Run the logging after authentication middleware/DRF authentication has resolved the user.",
            "Check why unauthenticated calls reach these endpoints (expired tokens correlate with 401 "
            "'Invalid token' responses).",
        ),
        "application",
    ),
    _rule(
        "django-field-error",
        r"Cannot resolve keyword '[^']+' into field",
        "An ORM query references a field that does not exist on the model (Django FieldError) - a code defect, "
        "typically a client-supplied filter or ordering parameter passed straight into .filter()/.order_by().",
        (
            "Whitelist allowed filter/ordering parameters per endpoint instead of forwarding request params.",
            "Return HTTP 400 for unknown fields rather than letting FieldError surface as a 500.",
            "Add a regression test for the failing endpoint and parameter.",
        ),
        "application",
    ),
    _rule(
        "jwt-auth-failure",
        r"could not validate credentials|invalid (?:access |auth )?token|invalid or expired auth code"
        r"|token (?:has )?expired|signature has expired"
        r"|not authenticated|HTTP/\d(?:\.\d)?\" 401",
        "Clients are presenting missing, expired or invalid authentication tokens.",
        (
            "Compare token lifetime with client refresh logic; refresh tokens before expiry.",
            "Verify server clock sync (chronyc tracking): clock skew breaks exp/nbf validation.",
            "Check whether a single client or IP dominates the failures (bug vs. credential stuffing).",
        ),
        "application",
    ),
    _rule(
        "authorization-denied",
        r"does not have permission|permission denied|forbidden|access denied|HTTP/\d(?:\.\d)?\" 403",
        "Authenticated users attempt actions their role does not allow (RBAC misconfiguration, or a UI that "
        "exposes restricted actions).",
        (
            "Break denials down by user and endpoint to separate misconfigured roles from misuse.",
            "Hide or disable UI actions the current role cannot perform.",
            "Review recent role and permission changes.",
        ),
        "application",
    ),
    _rule(
        "segfault",
        r"segfault|segmentation fault|core dumped|general protection fault",
        "A native process crashed (memory corruption, incompatible library, or hardware issue).",
        (
            "Collect the core dump (coredumpctl list / coredumpctl info <pid>) and the faulting library.",
            "Check for recent package upgrades and roll back if correlated.",
        ),
        "service owner",
    ),
    _rule(
        "systemd-unit-failure",
        r"failed with result|entered failed state|start request repeated too quickly|failed to start",
        "A systemd unit failed to start or crashed repeatedly.",
        (
            "Inspect the unit: systemctl status <unit>; journalctl -u <unit> -n 200 --no-pager.",
            "Fix configuration or dependencies, then systemctl reset-failed <unit> && systemctl restart <unit>.",
            "Set Restart=on-failure with RestartSec to avoid start-limit lockouts.",
        ),
        "service owner",
    ),
    _rule(
        "python-runtime-error",
        r"indices must be integers|TypeError|AttributeError|KeyError|NoneType|not subscriptable"
        r"|unexpected keyword argument|ValueError",
        "Application code defect: a Python exception caused by unexpected data shapes (for example indexing "
        "bytes or str with a string key because a response body was not JSON-decoded).",
        (
            "Log the full traceback (logger.exception) at the failing call site to pinpoint the line.",
            "Validate and parse external responses before indexing them (resp.json() / json.loads).",
            "Add a regression test that reproduces the payload, then deploy the fix.",
        ),
        "application",
    ),
    _rule(
        "upstream-unavailable",
        r"connection refused|service unavailable|bad gateway|gateway time-?out|upstream (?:timed out|prematurely"
        r" closed)|no route to host|connection timed out|failed to connect|HTTP/\d(?:\.\d)?\" 50[234]",
        "A dependency (upstream service, database or external API) is unreachable or overloaded.",
        (
            "Check the dependency's health and recent deploys; verify DNS and security-group rules.",
            "Add client timeouts, retries with exponential backoff and a circuit breaker.",
            "Correlate with the dependency's own logs for the same time window.",
        ),
        "service owner",
    ),
    _rule(
        "process-restart",
        r"\bpid \d+ last seen .* first seen",
        "The process was restarted: a deployment, a supervisor restart after a crash, or a crash loop.",
        (
            "Check whether restarts coincide with deployments (CI/CD history) or follow errors/OOM kills.",
            "Inspect the supervisor: journalctl -u <service> --since '<time>' or the process manager logs.",
            "If unplanned, enable crash reporting (core dumps, Sentry) and alert on restart frequency.",
        ),
        "service owner",
    ),
)


# Assigned explicitly to volume-spike incidents (never matched from text).
LOG_FLOOD_RULE = KnowledgeRule(
    "log-flood",
    _TITLES["log-flood"],
    re.compile(r"(?!x)x"),
    "A burst of logging dominated by a few statements (see evidence): typically a batch job, a retry loop or "
    "verbose debug output (payload dumps, banners) left enabled in production - not necessarily a failure, but "
    "it inflates storage cost and buries real errors.",
    (
        "Identify the emitting code paths from the dominant templates in the evidence.",
        "Demote payload dumps and banner prints to DEBUG and run production at INFO or WARNING.",
        "Use structured logging with sampling or rate limits for hot loops.",
    ),
    "application",
)


class KnowledgeBase:
    def __init__(self, rules: tuple[KnowledgeRule, ...] = DEFAULT_RULES) -> None:
        self._rules = rules

    def match(self, text: str) -> KnowledgeRule | None:
        return next((rule for rule in self._rules if rule.pattern.search(text)), None)

    def get(self, name: str) -> KnowledgeRule | None:
        if name == LOG_FLOOD_RULE.name:
            return LOG_FLOOD_RULE
        return next((rule for rule in self._rules if rule.name == name), None)
