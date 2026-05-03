"""Sample incoming vendor contracts for the contract compliance demo.

Three agreements with a deliberate mix of compliant and non-compliant clauses
so the compliance-check skill produces findings across every severity level.

Expected findings summary
─────────────────────────
contract-001  Cloud Software Services Agreement
  liability         — NON-COMPLIANT (high)    cap is 3 months of fees, rule requires 12 months
  consequential     — NON-COMPLIANT (high)    exclusion is one-sided (Vendor only)
  data_breach       — NON-COMPLIANT (medium)  48-hour notification, rule requires 24 hours
  payment           — COMPLIANT               net-30
  termination       — COMPLIANT               90-day convenience notice
  indemnification   — COMPLIANT               mutual obligations

contract-002  Professional Services Master Agreement
  liability         — NON-COMPLIANT (high)    cap is 6 months of fees, rule requires 12 months
  payment           — NON-COMPLIANT (high)    net-15, rule requires minimum net-30
  late_interest     — NON-COMPLIANT (medium)  2 % per month, rule allows max 1.5 %
  termination       — NON-COMPLIANT (high)    14-day notice, rule requires minimum 30 days
  governing_law     — NON-COMPLIANT (medium)  Delaware, rule requires New York
  confidentiality   — NON-COMPLIANT (medium)  1-year term, rule requires minimum 3 years
  indemnification   — NON-COMPLIANT (high)    Company indemnifies Vendor for Vendor's IP claims

contract-003  Data Analytics Platform Agreement
  liability         — COMPLIANT               12-month fee cap, mutual consequential exclusion
  payment           — COMPLIANT               net-45
  termination       — COMPLIANT               60-day convenience notice
  governing_law     — COMPLIANT               New York
  confidentiality   — COMPLIANT               3-year term
  data_breach       — NON-COMPLIANT (critical) 72-hour notification, rule requires 24 hours
  subprocessors     — NON-COMPLIANT (high)    Vendor may add without prior Company authorisation
  ip_ownership      — NON-COMPLIANT (high)    deliverables owned by Vendor, Company gets licence only
"""

from cogbase.core.models import Document

_META = {"doc_type": "contract"}

CONTRACTS_DOCUMENTS: list[Document] = [

    # ------------------------------------------------------------------
    # contract-001 — Cloud Software Services Agreement
    # Issues: liability cap (3 months), one-sided consequential exclusion,
    #         48-hour breach notification
    # ------------------------------------------------------------------
    Document(
        doc_id="contract-001",
        metadata={**_META, "source": "apex_cloud_saas_agreement.txt"},
        text="""\
CLOUD SOFTWARE SERVICES AGREEMENT

This Cloud Software Services Agreement ("Agreement") is entered into as of
March 1, 2025 ("Effective Date") by and between:

  Apex Cloud Solutions Inc., a Delaware corporation with its principal place
  of business at 400 Technology Park, San Jose, CA 95110 ("Vendor"),

and

  Acme Corporation, a New York corporation with its principal place of
  business at 100 Main Street, New York, NY 10001 ("Company").

RECITALS

WHEREAS, Vendor provides cloud-based project management and collaboration
software as a service; and

WHEREAS, Company desires to subscribe to Vendor's software services on the
terms set forth herein.

NOW, THEREFORE, the parties agree as follows:

ARTICLE 1 — SERVICES AND SUBSCRIPTION

1.1  Vendor shall make available to Company the cloud software platform
described in Exhibit A (the "Services") during the Subscription Term.

1.2  The initial Subscription Term begins on the Effective Date and continues
for twenty-four (24) months, expiring on February 28, 2027, unless earlier
terminated in accordance with this Agreement.

1.3  The annual subscription fee is USD 180,000, invoiced in advance on the
first day of each contract year.

ARTICLE 2 — PAYMENT TERMS

2.1  Company shall pay each undisputed invoice within thirty (30) days of
receipt ("Payment Due Date").

2.2  Invoices must be submitted electronically to accounts-payable@company.com
and must include the purchase order number, itemised service description, and
applicable taxes.

2.3  Amounts not paid by the Payment Due Date shall accrue interest at a rate
of 1.5% per month on the outstanding balance until paid in full.

ARTICLE 3 — INTELLECTUAL PROPERTY

3.1  The Services and all related software, documentation, and materials are
and remain the exclusive property of Vendor.  No title or ownership is
transferred to Company.

3.2  Vendor grants Company a non-exclusive, non-transferable licence to access
and use the Services solely for Company's internal business purposes during
the Subscription Term.

3.3  All data submitted by Company to the Services ("Company Data") remains
the exclusive property of Company.  Vendor shall not use Company Data for
any purpose other than providing the Services.

ARTICLE 4 — CONFIDENTIALITY

4.1  Each party agrees to hold the other's confidential information in strict
confidence and to use it only for the purposes of this Agreement.

4.2  The confidentiality obligations under this Article shall survive for a
period of three (3) years following the expiration or termination of this
Agreement.

4.3  Standard exceptions to confidentiality apply: information that is or
becomes publicly available through no breach of this Agreement; information
independently developed; and information required to be disclosed by law or
court order.

ARTICLE 5 — LIMITATION OF LIABILITY

5.1  VENDOR'S TOTAL AGGREGATE LIABILITY TO COMPANY FOR ALL CLAIMS ARISING
UNDER OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED THE TOTAL FEES PAID BY
COMPANY TO VENDOR IN THE THREE (3) MONTHS IMMEDIATELY PRECEDING THE EVENT
GIVING RISE TO THE CLAIM.

5.2  IN NO EVENT SHALL VENDOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL,
PUNITIVE, OR CONSEQUENTIAL DAMAGES, INCLUDING LOSS OF PROFITS, LOSS OF DATA,
OR BUSINESS INTERRUPTION, REGARDLESS OF THE THEORY OF LIABILITY AND EVEN IF
VENDOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

5.3  The limitations in Sections 5.1 and 5.2 do not apply to (a) death or
personal injury caused by negligence, (b) fraud, or (c) Vendor's
indemnification obligations under Article 6.

ARTICLE 6 — INDEMNIFICATION

6.1  Each party (the "Indemnifying Party") shall indemnify, defend, and hold
harmless the other party and its officers, directors, and employees from
any third-party claims, damages, losses, and reasonable legal fees arising
from the Indemnifying Party's (a) breach of this Agreement, (b) negligence
or wilful misconduct, or (c) infringement of any third-party intellectual
property right.

6.2  The indemnified party must promptly notify the indemnifying party of the
claim, grant sole control of the defence, and reasonably cooperate.

ARTICLE 7 — DATA PROTECTION AND SECURITY

7.1  Vendor shall implement appropriate technical and organisational measures
to protect Company Data against unauthorised access, disclosure, or loss,
including encryption at rest and in transit.

7.2  In the event of a confirmed or suspected personal data breach affecting
Company Data, Vendor shall notify Company by email to
security-incidents@company.com within forty-eight (48) hours of becoming
aware of such breach, and shall provide the information required by applicable
data protection law.

7.3  Upon expiration or termination of this Agreement, Vendor shall securely
delete all Company Data within thirty (30) days and provide written
certification of deletion.

ARTICLE 8 — TERMINATION

8.1  Either party may terminate this Agreement for convenience upon ninety
(90) days' prior written notice to the other party.

8.2  Either party may terminate immediately upon written notice if the other
party materially breaches this Agreement and fails to cure such breach within
thirty (30) days of written notice describing the breach.

8.3  Either party may terminate immediately if the other party becomes
insolvent, is adjudicated bankrupt, or makes an assignment for the benefit
of creditors.

8.4  Upon termination for any reason, Vendor shall provide reasonable
transition assistance for up to sixty (60) days at the then-current time and
materials rates.

ARTICLE 9 — GOVERNING LAW AND DISPUTE RESOLUTION

9.1  This Agreement shall be governed by the laws of the State of New York,
without regard to conflict-of-law principles.

9.2  Any dispute not resolved by the parties within thirty (30) days of
written notice shall be submitted to binding arbitration under the Commercial
Arbitration Rules of the American Arbitration Association, with venue in
New York City.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the
Effective Date.

APEX CLOUD SOLUTIONS INC.              ACME CORPORATION
By: ______________________             By: ______________________
Name:                                  Name:
Title:                                 Title:
Date:                                  Date:
""",
    ),

    # ------------------------------------------------------------------
    # contract-002 — Professional Services Master Agreement
    # Issues: 6-month liability cap, net-15 payment, 2%/month late interest,
    #         14-day termination notice, Delaware law, 1-year confidentiality,
    #         one-sided indemnification for Vendor's IP
    # ------------------------------------------------------------------
    Document(
        doc_id="contract-002",
        metadata={**_META, "source": "meridian_consulting_psma.txt"},
        text="""\
PROFESSIONAL SERVICES MASTER AGREEMENT

This Professional Services Master Agreement ("Agreement") is made effective
as of June 1, 2025 ("Effective Date") between:

  Meridian Consulting Group LLC, a Delaware limited liability company with
  offices at 88 Commerce Drive, Wilmington, DE 19801 ("Vendor"),

and

  Acme Corporation, a New York corporation at 100 Main Street,
  New York, NY 10001 ("Company").

ARTICLE 1 — SCOPE OF SERVICES

1.1  Vendor will perform consulting, analysis, and advisory services as
described in individual Statements of Work ("SOW") executed by both parties
and incorporated by reference into this Agreement.

1.2  Each SOW will specify deliverables, timelines, fees, and any special
terms applicable to that engagement.

1.3  The initial term of this Agreement begins on the Effective Date and
continues for twelve (12) months, expiring on May 31, 2026.  The Agreement
shall automatically renew for successive one-year terms unless either party
provides written notice of non-renewal at least thirty (30) days before
the end of the then-current term.

ARTICLE 2 — FEES AND PAYMENT

2.1  Vendor will invoice Company monthly in arrears for services rendered
in the preceding calendar month, together with reimbursable expenses
pre-approved in writing by Company.

2.2  Company shall pay all undisputed invoices within fifteen (15) days of
receipt.

2.3  Any amounts not paid within the fifteen-day period shall be subject to
a late payment charge of two percent (2%) per month, compounded monthly,
on all overdue amounts from the due date until the date of payment.

2.4  Vendor reserves the right to suspend services without liability if any
invoice remains unpaid for more than thirty (30) days after the due date.

ARTICLE 3 — INTELLECTUAL PROPERTY

3.1  All work product, deliverables, reports, analyses, methodologies, and
materials created by Vendor in the performance of services under this
Agreement ("Deliverables") shall remain the sole and exclusive property of
Vendor.

3.2  Vendor grants Company a perpetual, non-exclusive, royalty-free licence
to use the Deliverables solely for Company's internal business purposes.

3.3  Vendor retains full ownership of all background intellectual property,
tools, methodologies, know-how, and frameworks used in delivery of the
services, regardless of whether they are incorporated into the Deliverables.

ARTICLE 4 — CONFIDENTIALITY

4.1  Each party shall protect the other's confidential information using the
same degree of care it uses to protect its own confidential information.

4.2  The obligations of confidentiality under this Article shall remain in
effect for a period of one (1) year following termination or expiration of
this Agreement.

4.3  Exceptions apply to information that is publicly known, independently
developed, or required by law to be disclosed.

ARTICLE 5 — LIMITATION OF LIABILITY

5.1  Each party's total aggregate liability to the other party for any and
all claims arising under or related to this Agreement shall not exceed the
total fees paid or payable to Vendor in the six (6) months immediately
preceding the event giving rise to the claim.

5.2  Neither party shall be liable for any indirect, incidental, special,
or consequential damages, including loss of profits or loss of data,
regardless of the theory of liability.

5.3  The foregoing limitations do not apply to death or personal injury
caused by negligence or to fraud.

ARTICLE 6 — INDEMNIFICATION

6.1  Vendor shall indemnify, defend, and hold harmless Company and its
affiliates from third-party claims arising from Vendor's (a) breach of this
Agreement, (b) negligence or wilful misconduct, or (c) infringement of any
third-party intellectual property right by any Deliverable.

6.2  Company shall indemnify, defend, and hold harmless Vendor and its
members, officers, and employees from any and all third-party claims,
damages, losses, and expenses (including legal fees) arising from or related
to (a) Company's breach of this Agreement, (b) Company's negligence or
wilful misconduct, and (c) any claim that materials, data, or information
provided by Company to Vendor infringe a third party's intellectual property
rights, including any claim that Vendor's use of such Company-provided
materials in performing the services infringes a third party's patent,
copyright, trade secret, or trademark.

ARTICLE 7 — DATA PROTECTION

7.1  Where Vendor processes personal data on behalf of Company, it shall
do so in accordance with applicable data protection laws and Company's
written instructions.

7.2  Vendor shall notify Company of any confirmed personal data breach
affecting Company's data within seventy-two (72) hours of becoming aware
of the breach.

7.3  Upon termination, Vendor shall return or destroy all Company personal
data within forty-five (45) days and certify such return or destruction.

ARTICLE 8 — TERMINATION

8.1  Either party may terminate this Agreement for convenience upon fourteen
(14) days' written notice to the other party.

8.2  Either party may terminate immediately for material breach if the
breaching party fails to cure the breach within twenty-one (21) days of
receiving written notice of the breach.

8.3  The Company may terminate immediately upon Vendor's insolvency or
bankruptcy.

ARTICLE 9 — GOVERNING LAW

9.1  This Agreement shall be governed by the laws of the State of Delaware,
without regard to its conflict-of-law provisions.

9.2  Any disputes shall be resolved in the state or federal courts located
in Wilmington, Delaware, and each party consents to personal jurisdiction
in such courts.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the
Effective Date.

MERIDIAN CONSULTING GROUP LLC          ACME CORPORATION
By: ______________________             By: ______________________
Name:                                  Name:
Title:                                 Title:
Date:                                  Date:
""",
    ),

    # ------------------------------------------------------------------
    # contract-003 — Data Analytics Platform Agreement
    # Issues: 72-hour breach notification, Vendor-side subprocessor freedom,
    #         IP ownership retained by Vendor
    # Compliant: 12-month liability cap, mutual consequential exclusion,
    #            net-45, 60-day termination, New York law, 3-year NDA
    # ------------------------------------------------------------------
    Document(
        doc_id="contract-003",
        metadata={**_META, "source": "datasphere_analytics_agreement.txt"},
        text="""\
DATA ANALYTICS PLATFORM AGREEMENT

This Data Analytics Platform Agreement ("Agreement") is entered into as of
September 1, 2025 ("Effective Date") between:

  DataSphere Analytics Corp., a California corporation with its principal
  offices at 2200 Innovation Way, Palo Alto, CA 94301 ("Vendor"),

and

  Acme Corporation, a New York corporation at 100 Main Street,
  New York, NY 10001 ("Company").

The total contract value for the initial term is USD 840,000.

ARTICLE 1 — PLATFORM AND SERVICES

1.1  Vendor shall provide Company with access to its proprietary data
analytics and business intelligence platform (the "Platform"), together with
implementation, configuration, and support services as described in Exhibit A.

1.2  The initial term is three (3) years commencing on the Effective Date
and expiring on August 31, 2028.

ARTICLE 2 — FEES AND PAYMENT

2.1  Company shall pay Vendor an annual platform licence fee of USD 280,000,
invoiced on the first day of each contract year.

2.2  All invoices are payable within forty-five (45) days of receipt.

2.3  Disputed invoices must be raised in writing within fifteen (15) business
days of receipt.  The undisputed portion is payable on the standard due date.

2.4  Vendor may increase the annual licence fee for each renewal year by no
more than the increase in the Consumer Price Index for the preceding twelve
months, provided at least ninety (90) days' written notice is given before
the renewal date.

ARTICLE 3 — INTELLECTUAL PROPERTY AND WORK PRODUCT

3.1  The Platform, including all software, algorithms, models, documentation,
and updates thereto, is and shall remain the sole and exclusive property of
Vendor.

3.2  All analyses, reports, dashboards, custom configurations, and other
materials specifically developed by Vendor for Company under this Agreement
("Custom Deliverables") shall be the sole and exclusive property of Vendor.
Vendor grants Company a perpetual, irrevocable, royalty-free, worldwide
licence to use, copy, and modify the Custom Deliverables for Company's
internal business purposes.

3.3  Company retains ownership of all Company Data loaded into the Platform.
Vendor shall not use Company Data for any purpose other than providing the
Platform services.

ARTICLE 4 — DATA PROTECTION AND SECURITY

4.1  Vendor shall process Company's personal data only on documented
instructions from Company and shall implement appropriate technical and
organisational safeguards, including AES-256 encryption at rest and in
transit.

4.2  Vendor shall enter into a Data Processing Agreement ("DPA") with
Company prior to any processing of personal data.

4.3  Vendor may engage subprocessors in the performance of services under
this Agreement without prior written consent from Company, provided that
Vendor (a) publishes an updated list of subprocessors on its website with
at least thirty (30) days' advance notice before any new subprocessor begins
processing Company's personal data, and (b) imposes data protection
obligations on each subprocessor equivalent to those in the DPA.

4.4  Vendor shall notify Company of any confirmed or reasonably suspected
personal data breach affecting Company's data as soon as reasonably
practicable and in any event no later than seventy-two (72) hours after
Vendor becomes aware of the breach.  Notification shall be made by email
to security-incidents@company.com and shall include: a description of the
nature of the breach; the categories and approximate number of records
affected; likely consequences; and measures taken to address the breach.

4.5  Vendor shall securely delete or return all Company Data within
thirty (30) days of expiration or termination of this Agreement and shall
provide written confirmation of deletion.

ARTICLE 5 — CONFIDENTIALITY

5.1  Each party shall hold the other's confidential information in strict
confidence using at least the same care it applies to its own confidential
information, but in no event less than reasonable care.

5.2  The confidentiality obligations under this Article shall remain in
effect for three (3) years following the termination or expiration of this
Agreement.  Obligations with respect to trade secrets shall survive
indefinitely.

5.3  Standard exceptions apply: publicly available information; independently
developed information; information required by law or court order to be
disclosed (with prompt written notice to the disclosing party).

ARTICLE 6 — LIMITATION OF LIABILITY

6.1  Each party's total aggregate liability for all claims arising under
or related to this Agreement shall not exceed the total fees paid by
Company to Vendor in the twelve (12) months immediately preceding the
event giving rise to the claim.

6.2  Neither party shall be liable to the other for any indirect, incidental,
special, punitive, or consequential damages, including loss of profits,
loss of revenue, loss of data, or business interruption, even if that party
has been advised of the possibility of such damages.  This mutual exclusion
applies regardless of the theory of liability.

6.3  Exceptions to the foregoing: the liability cap and consequential-damages
exclusion do not apply to (a) death or personal injury caused by negligence,
(b) fraud, (c) either party's indemnification obligations, or (d) breaches
of confidentiality or data protection obligations.

ARTICLE 7 — INDEMNIFICATION

7.1  Vendor shall indemnify, defend, and hold harmless Company from
third-party claims arising from Vendor's (a) breach of this Agreement,
(b) negligence or wilful misconduct, or (c) infringement of a third-party
intellectual property right by the Platform or any Vendor-developed material.

7.2  Company shall indemnify, defend, and hold harmless Vendor from
third-party claims arising from Company's (a) breach of this Agreement or
(b) negligence or wilful misconduct.

7.3  Indemnification procedure: the indemnified party must promptly notify
the indemnifying party of the claim, grant sole control of the defence, and
cooperate reasonably.  The indemnifying party may not settle any claim that
imposes liability on the indemnified party without prior written consent.

ARTICLE 8 — TERMINATION

8.1  Either party may terminate this Agreement for convenience upon sixty
(60) days' prior written notice.

8.2  Either party may terminate for cause if the other party materially
breaches this Agreement and fails to cure such breach within thirty (30)
days of receiving written notice identifying the breach.

8.3  Either party may terminate immediately upon the other party's insolvency,
bankruptcy, or cessation of business.

8.4  The Company may terminate immediately if Vendor suffers a security
incident that materially compromises Company Data, or if Vendor undergoes a
change of control to a direct competitor of Company.

8.5  Upon termination, Vendor shall provide transition assistance for up to
ninety (90) days at pre-agreed rates.

ARTICLE 9 — GOVERNING LAW AND DISPUTE RESOLUTION

9.1  This Agreement shall be governed by the laws of the State of New York,
without regard to conflict-of-law principles.

9.2  The parties shall attempt to resolve disputes through senior management
escalation for thirty (30) days following written notice of the dispute.

9.3  Disputes not resolved through escalation shall be submitted to binding
arbitration under the Commercial Arbitration Rules of the American
Arbitration Association, with venue in New York City, New York.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the
Effective Date.

DATASPHERE ANALYTICS CORP.             ACME CORPORATION
By: ______________________             By: ______________________
Name:                                  Name:
Title:                                 Title:
Date:                                  Date:
""",
    ),
]
