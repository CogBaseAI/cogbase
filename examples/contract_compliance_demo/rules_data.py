"""Sample company rule documents for the contract compliance demo.

Five policy documents covering the clause types most commonly reviewed in
vendor agreements.  Each document is chunked and embedded into ``rule_chunks``
during ingestion.  The LLM judge retrieves the top matching chunks for each
contract clause and uses them as the authoritative company policy.
"""

from cogbase.core.models import Document

_META = {
    "doc_type": "rules",
    "source": "vendor_contract_standards.txt",
    "ruleset_id": "company_rules_v1",
}

RULES_DOCUMENTS: list[Document] = [

    # ------------------------------------------------------------------
    # rules-001 — Liability and Indemnification
    # ------------------------------------------------------------------
    Document(
        doc_id="rules-001",
        metadata={**_META, "topic": "liability_indemnification"},
        text="""\
COMPANY VENDOR CONTRACT STANDARDS — LIABILITY AND INDEMNIFICATION
Policy version: 1.4  |  Effective: 2024-01-01  |  Owner: Legal

1. LIABILITY CAP

1.1  In all vendor agreements the total aggregate liability of either party
for any and all claims arising under or related to the agreement shall not
exceed the greater of (a) the total fees paid or payable by the Company to
the Vendor in the twelve (12) months immediately preceding the event giving
rise to the claim, or (b) USD 500,000.

1.2  The liability cap applies to all theories of liability, including
contract, tort (including negligence), strict liability, and breach of
statutory duty, regardless of whether the party has been advised of the
possibility of such damages.

1.3  Exceptions to the liability cap: the cap does not apply to (a) death
or personal injury caused by negligence, (b) fraud or fraudulent
misrepresentation, (c) the Vendor's indemnification obligations under
Section 2, or (d) breaches of confidentiality or data protection obligations.

2. EXCLUSION OF CONSEQUENTIAL DAMAGES

2.1  Neither party shall be liable to the other for any indirect,
incidental, special, punitive, or consequential damages, including loss of
profits, loss of revenue, loss of data, or business interruption, even if
that party has been advised of the possibility of such damages.

2.2  The exclusion of consequential damages must be mutual.  Agreements
that exclude consequential damages only for the Vendor while preserving them
for the Company are not acceptable.

3. INDEMNIFICATION

3.1  The Vendor shall indemnify, defend, and hold harmless the Company and
its officers, directors, employees, and agents from any third-party claims,
damages, losses, and expenses (including reasonable legal fees) arising from:
  (a) the Vendor's breach of any representation, warranty, or obligation
      under the agreement;
  (b) the Vendor's negligence or wilful misconduct;
  (c) infringement of any third-party intellectual property right by the
      Vendor's deliverables or services.

3.2  The Company shall indemnify, defend, and hold harmless the Vendor from
third-party claims arising from the Company's breach of the agreement or the
Company's negligence or wilful misconduct.

3.3  Indemnification obligations must be mutual.  Agreements that require
the Company to indemnify the Vendor for the Vendor's own intellectual
property infringement, or for claims arising from the Vendor's acts, are not
acceptable.

3.4  The indemnified party must (a) promptly notify the indemnifying party
of the claim in writing, (b) grant the indemnifying party sole control of
the defence and settlement (subject to the indemnified party's right to
participate at its own expense), and (c) reasonably cooperate with the
indemnifying party.

3.5  The indemnifying party may not settle any claim that imposes liability
or obligations on the indemnified party without prior written consent.
""",
    ),

    # ------------------------------------------------------------------
    # rules-002 — Payment Terms
    # ------------------------------------------------------------------
    Document(
        doc_id="rules-002",
        metadata={**_META, "topic": "payment_terms"},
        text="""\
COMPANY VENDOR CONTRACT STANDARDS — PAYMENT TERMS AND INVOICING
Policy version: 2.1  |  Effective: 2024-01-01  |  Owner: Finance / Legal

1. STANDARD PAYMENT TERMS

1.1  The Company's standard payment term is net-45 (45 days) from receipt
of a valid invoice.  Payment terms shorter than net-30 are not acceptable
without written approval from the CFO.

1.2  Invoices must be submitted electronically to accounts-payable@company.com
and must include: (a) invoice number, (b) purchase order number,
(c) itemised description of goods or services, (d) unit prices and quantities,
(e) applicable taxes, and (f) Vendor's bank details.

1.3  Invoices that do not comply with Section 1.2 will be rejected and the
payment period will not start until a compliant invoice is received.

2. LATE PAYMENT INTEREST

2.1  The Company will not accept late-payment interest clauses that exceed
the lower of (a) 1.5% per month on overdue amounts, or (b) the maximum rate
permitted by applicable law.

2.2  Late-payment interest may only begin to accrue after the Vendor has
notified the Company in writing of the overdue invoice and provided a
grace period of at least ten (10) business days for the Company to cure
the non-payment.

3. DISPUTED INVOICES

3.1  If the Company disputes an invoice in good faith, it shall notify the
Vendor in writing within fifteen (15) business days of invoice receipt,
identifying the disputed amount and the basis for the dispute.

3.2  The Company shall pay the undisputed portion of the invoice by the
due date.  Disputed amounts are held pending resolution and do not accrue
late-payment interest while the dispute is being resolved in good faith.

4. PRICE CHANGES

4.1  Vendors may not increase fees during the initial contract term without
prior written agreement from the Company.

4.2  Price increases for renewal terms must be notified in writing at least
ninety (90) days before the renewal date and shall not exceed 5% above the
Consumer Price Index for the previous twelve months unless separately agreed.

5. TAXES

5.1  All fees quoted in the agreement are exclusive of applicable sales,
use, value-added, or withholding taxes unless explicitly stated otherwise.

5.2  Each party is responsible for its own income taxes.  If the Company is
required by law to withhold taxes on payments to the Vendor, the Company
shall deduct the applicable withholding and remit to the relevant authority.
The net payment to the Vendor will be reduced accordingly unless the Vendor
provides a valid exemption certificate.
""",
    ),

    # ------------------------------------------------------------------
    # rules-003 — Data Privacy and Security
    # ------------------------------------------------------------------
    Document(
        doc_id="rules-003",
        metadata={**_META, "topic": "data_privacy_security"},
        text="""\
COMPANY VENDOR CONTRACT STANDARDS — DATA PRIVACY AND SECURITY
Policy version: 3.0  |  Effective: 2024-01-01  |  Owner: CISO / Legal

1. DATA PROTECTION OBLIGATIONS

1.1  Vendors that process personal data on behalf of the Company must enter
into a Data Processing Agreement (DPA) that complies with the General Data
Protection Regulation (GDPR) and all other applicable data protection laws.

1.2  The Vendor must process personal data only on documented instructions
from the Company and for no other purpose.

1.3  The Vendor must implement and maintain appropriate technical and
organisational measures to protect personal data against accidental or
unlawful destruction, loss, alteration, unauthorised disclosure, or access.
At a minimum this includes encryption of personal data at rest and in
transit using industry-standard algorithms (AES-256 or equivalent).

2. SUBPROCESSORS

2.1  The Vendor must not engage any subprocessor to process the Company's
personal data without prior written authorisation from the Company.

2.2  Where authorisation is granted, the Vendor must impose on the
subprocessor data protection obligations equivalent to those in the DPA.

2.3  The Vendor remains fully liable to the Company for the acts and
omissions of its subprocessors.

3. DATA BREACH NOTIFICATION

3.1  The Vendor must notify the Company of any confirmed or reasonably
suspected personal data breach within 24 hours of the Vendor becoming
aware of it.  Notification must be made by email to
security-incidents@company.com and by telephone to the Company's CISO.

3.2  Notification must include, to the extent known: (a) a description of
the nature of the breach, (b) the categories and approximate number of
individuals and records concerned, (c) likely consequences of the breach,
and (d) measures taken or proposed to address the breach.

3.3  Breach notification timelines longer than 72 hours are not acceptable.
Contracts that permit notification "as soon as reasonably practicable"
without a defined maximum period are not acceptable.

4. DATA RETENTION AND DELETION

4.1  The Vendor must not retain the Company's personal data or confidential
data beyond the period necessary to perform the contracted services.

4.2  Upon expiry or termination of the agreement, the Vendor must securely
delete or return all the Company's data within 30 days, and certify in
writing that deletion is complete.

5. SECURITY ASSESSMENTS

5.1  The Company reserves the right to conduct or commission security
assessments, audits, or penetration tests of the Vendor's systems that
process the Company's data, upon 14 days' written notice.

5.2  The Vendor must promptly remediate any critical or high-severity
vulnerabilities identified in such assessments within 30 days, and provide
written confirmation of remediation.
""",
    ),

    # ------------------------------------------------------------------
    # rules-004 — Termination and Dispute Resolution
    # ------------------------------------------------------------------
    Document(
        doc_id="rules-004",
        metadata={**_META, "topic": "termination_dispute_resolution"},
        text="""\
COMPANY VENDOR CONTRACT STANDARDS — TERMINATION AND DISPUTE RESOLUTION
Policy version: 1.6  |  Effective: 2024-01-01  |  Owner: Legal

1. TERMINATION FOR CONVENIENCE

1.1  The Company must retain the right to terminate any vendor agreement for
convenience upon written notice.  The minimum acceptable notice period for
termination for convenience is 30 days for agreements valued below USD 250,000
per year, and 60 days for agreements valued at or above USD 250,000 per year.

1.2  Agreements that require the Company to pay a termination fee or penalty
for exercising its termination-for-convenience right are not acceptable unless
the fee is (a) reasonable relative to the Vendor's actual unrecoverable costs
and (b) capped at no more than three months of the applicable fees.

2. TERMINATION FOR CAUSE

2.1  Either party must have the right to terminate immediately for material
breach if the breach remains uncured for 30 days after written notice.

2.2  Either party must have the right to terminate immediately (without cure
period) upon the other party's insolvency, bankruptcy, assignment for the
benefit of creditors, or cessation of business operations.

2.3  The Company must have the right to terminate immediately if the Vendor
(a) suffers a security incident that materially affects the Company's data,
(b) violates applicable law in connection with services provided to the Company,
or (c) undergoes a change of control to a direct competitor of the Company.

3. EFFECTS OF TERMINATION

3.1  Upon termination for any reason, the Vendor must (a) immediately cease
all use of the Company's confidential information and data, (b) return or
destroy Company data as specified in the data protection standards, and
(c) provide reasonable transition assistance for up to 90 days at the
Company's request and at pre-agreed rates.

3.2  Provisions that survive termination must be explicitly listed in the
agreement.  Provisions that typically survive include: confidentiality,
data protection, IP ownership, limitation of liability, indemnification,
and accrued payment obligations.

4. DISPUTE RESOLUTION

4.1  The governing law of all vendor agreements shall be the laws of the
State of New York, United States, unless the Vendor is incorporated outside
the United States, in which case New York law is still strongly preferred.

4.2  Before initiating formal proceedings, the parties must attempt to resolve
disputes through senior management escalation for a period of 30 days following
written notice of the dispute.

4.3  Any dispute not resolved through senior management escalation shall be
submitted to binding arbitration administered by the American Arbitration
Association under its Commercial Arbitration Rules.  The number of arbitrators
shall be one for disputes below USD 1 million and three for disputes at or above
USD 1 million.

4.4  Venue for arbitration shall be New York City, New York.

4.5  Each party shall bear its own legal fees and expenses in arbitration
unless the arbitrator(s) determine that a party acted in bad faith.

4.6  Agreements that require disputes to be resolved exclusively in the courts
of another country, or that specify a governing law other than New York, require
written approval from the General Counsel.
""",
    ),

    # ------------------------------------------------------------------
    # rules-005 — Intellectual Property and Confidentiality
    # ------------------------------------------------------------------
    Document(
        doc_id="rules-005",
        metadata={**_META, "topic": "ip_confidentiality"},
        text="""\
COMPANY VENDOR CONTRACT STANDARDS — INTELLECTUAL PROPERTY AND CONFIDENTIALITY
Policy version: 2.3  |  Effective: 2024-01-01  |  Owner: Legal

1. OWNERSHIP OF WORK PRODUCT

1.1  All deliverables, work product, inventions, developments, software, and
other materials created by the Vendor specifically for the Company under the
agreement, and paid for by the Company, shall be owned exclusively by the
Company as works-made-for-hire to the fullest extent permitted by law.

1.2  To the extent any deliverable does not qualify as a work-made-for-hire,
the Vendor hereby assigns to the Company all right, title, and interest in and
to such deliverable, including all intellectual property rights therein.

1.3  The Vendor retains ownership of its pre-existing intellectual property
("Background IP"), including tools, methodologies, frameworks, and libraries
that existed before the commencement of the agreement.

1.4  The Vendor grants the Company a perpetual, irrevocable, royalty-free,
worldwide licence to use any Background IP that is embedded in or necessary
to use the deliverables.

1.5  Agreements that attempt to grant the Vendor ownership or a licence-back
to custom deliverables paid for by the Company are not acceptable.

2. LICENCE GRANTS

2.1  Where the agreement involves the Company licensing software or a service
from the Vendor, the licence must be (a) non-exclusive, (b) limited to the
scope described in the agreement, and (c) subject to the payment obligations.

2.2  The Vendor must not include "click-wrap" or "shrink-wrap" terms that
modify or override the negotiated agreement without the Company's prior
written consent.

3. CONFIDENTIALITY

3.1  Each party must treat the other's confidential information with at least
the same degree of care it uses to protect its own confidential information,
and in any case with no less than reasonable care.

3.2  The standard confidentiality term must be no shorter than three (3) years
from the date of disclosure.  Confidentiality obligations for trade secrets
must survive indefinitely.

3.3  Standard exceptions to confidentiality obligations include: (a) information
that was already publicly known at the time of disclosure, (b) information
that becomes publicly known through no fault of the receiving party,
(c) information independently developed by the receiving party without use of
the disclosing party's confidential information, and (d) information required
to be disclosed by law or court order, provided the receiving party gives
prompt written notice to enable the disclosing party to seek a protective order.

3.4  The Vendor must not (a) disclose the existence or terms of the agreement
to third parties without consent, or (b) use the Company's name, logo, or
trademarks in marketing materials without prior written approval.

4. OPEN-SOURCE SOFTWARE

4.1  The Vendor must disclose all open-source components incorporated into any
deliverable prior to delivery.

4.2  The Vendor must not incorporate open-source software licensed under the
GNU General Public License (GPL), the GNU Affero General Public License (AGPL),
or any other "copyleft" licence that could require the Company to open-source
its proprietary code, without prior written approval from the General Counsel.
""",
    ),
]
