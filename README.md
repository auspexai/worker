# AuspexAI Worker

The volunteer worker process for [AuspexAI](https://github.com/auspexai) — runs on volunteer machines, executes work units dispatched by the coordinator, donates compute to AI safety research.

## Status

**Phase 0 — Foundation.** Code begins in Phase 1.

## Scope

The Worker:

- Runs on volunteer machines (macOS Intel/ARM, Linux x86_64/ARM64; Windows under consideration)
- Generates an Ed25519 keypair on first run, stored in OS-native keystore (Keychain / DPAPI / libsecret) — volunteers never paste keys into web forms
- Enrolls anonymously (T0) or upgrades to verified identity via OAuth 2.0 Device Authorization Flow (T1+)
- Executes work units within sandboxed limits (CPU, GPU, RAM, network caps; idle-only mode by default)
- Submits signed results
- Maintains a local audit log of what it ran, when, and for which job

Worker trust tiers (T0 anonymous → T4 maintainer) govern work eligibility and quorum weight; see the AuspexAI Principles & Scope §6 for the trust model.

## Conduct on the network

Worker conduct on the AuspexAI network is governed by the **Volunteer Terms of Participation** (forthcoming, Phase 2). Network-level abuse — running malicious workers, abusing reputation systems, attempting to influence quorum — is handled through technical enforcement (revocation, ban, trust-tier demotion), not Code-of-Conduct enforcement.

## License

[AGPL-3.0](LICENSE) — workers are network-served clients of the AuspexAI coordinator and inherit the platform's copyleft posture.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) (org-wide). Worker contributions follow the standard PR workflow with DCO sign-off; substantial architectural contributions require an RFC before code is written.

## Governance

Project direction is held by the Maintainer team per [`GOVERNANCE.md`](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md). Code of Conduct: [`CODE_OF_CONDUCT.md`](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md).

## Watch this repo

Activity will begin as Phase 1 ramps up.
