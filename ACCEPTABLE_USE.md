# Acceptable Use Policy

VOX is a singing voice conversion engine. Voice synthesis is a dual-use
technology: the same model that helps a person sing in their own voice can
also be misused to impersonate someone else. This document lists what we
consider acceptable use, what we do not, and the responsibilities of anyone
who runs or distributes this code.

Apache-2.0 grants you broad rights. **This policy describes the conditions
under which the authors believe the technology is being used responsibly.
Violation does not revoke the Apache license, but it does revoke the
authors' endorsement, support, and willingness to help with downstream
issues** — and may violate the law in your jurisdiction.

---

## 1. Acceptable use

You **may** use VOX to:

- Train and synthesise **your own voice** for personal or professional
  creative work (covers, original songs, demos, livestream, education).
- Train another person's voice **only with their explicit, informed,
  documented consent** for that specific use case. Past general consent
  does not cover voice cloning.
- Build downstream tools, plugins, or services on top of VOX, provided
  those downstream products carry forward this Acceptable Use Policy or a
  stricter equivalent.
- Use VOX in academic research, conference demos, or technical writing,
  with attribution to this repository.
- Use VOX outputs in **fiction, parody, and clearly labelled creative
  works** where the AI origin is disclosed.

## 2. Prohibited use

You **must not** use VOX to:

- Clone the voice of a real person **without their explicit informed
  consent**, including but not limited to: celebrities, politicians,
  journalists, family members, ex-partners, classmates, colleagues,
  customer-service workers, or any minor.
- Create audio that could plausibly be mistaken for a real recording of a
  real person, **for fraud, harassment, defamation, sexual content,
  political manipulation, or any unlawful purpose**.
- Bypass voice-authentication systems (banking IVR, smart home, etc.).
- Generate audio of a real minor in any voice or context.
- Distribute trained model checkpoints of an identifiable person's voice
  to third parties without that person's documented consent for
  redistribution (consent to use ≠ consent to share the model).
- Train on copyrighted audio without a licence (e.g. ripping commercial
  vocals or anime character voices and publishing the resulting model).
- Misrepresent VOX-generated audio as a human performance in any context
  where authenticity is material (auditions, voice acting credit,
  certification recordings, court evidence, etc.).
- Use VOX outputs to harass, intimidate, or threaten any person.

## 3. Required practices when releasing outputs

If you publish, distribute, or commercially release VOX-generated audio:

1. **Label it** as AI-generated or AI-assisted in the description,
   metadata, or release notes.
2. **Credit** the original singer if the source vocal was someone else's
   performance.
3. **Resolve rights** for any underlying composition (mechanical /
   synchronisation / publishing) through the appropriate route
   (DistroKid, JASRAC, direct licence, public domain confirmation, etc.).
4. **Keep an audit trail** of consent for any non-self voice used,
   sufficient to defend against a takedown request.

## 4. Required practices when redistributing models

If you publish a trained VOX checkpoint:

1. Attach a **MODEL_CARD.md** (see template in this repository) that names
   the dataset, training duration, intended use, and known limitations.
2. State the **consent basis** for the voice in the dataset.
3. Carry forward this Acceptable Use Policy or a stricter equivalent.
4. Do not strip provenance from the model files (do not rename / remove
   metadata that identifies the source repository).

## 5. Responsibilities

| Role | What you are responsible for |
|---|---|
| You, the user | Compliance with this policy, with local law, and with consent obligations to anyone whose voice you process. |
| You, if you redistribute | Carrying forward this policy and the model card. Vetting downstream users where feasible. |
| The authors | Maintaining this code under Apache-2.0. We do not warrant your specific use. We are not your lawyer. |

## 6. Reporting misuse

If you become aware of VOX being used in violation of this policy:

- Open a non-sensitive issue describing the pattern at
  https://github.com/ashina814/vox-engine/issues
- For incidents that involve a real person's privacy, safety, or
  identity, contact the repository owner directly rather than opening a
  public issue.

## 7. Why this matters

The authors built VOX expecting it to be used for self-expression — including
by people who cannot sing in their own voice because of injury, transition,
age, or other circumstances. We released it openly because we believe that
empowerment is worth the risk of misuse, **provided users take the safety
practices above seriously**. If you find this policy too restrictive for
your intended use, please reconsider whether VOX is the right tool.

---

*This policy may be updated as legal and social norms around AI voice
synthesis evolve (e.g. EU AI Act, national deepfake regulation). Major
changes will be announced in repository release notes.*

*Document Version: 1.0 / Date: 2026-05-20*
