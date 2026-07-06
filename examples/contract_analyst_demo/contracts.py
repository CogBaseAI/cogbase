"""Realistic legal-contract fixtures for the Contract Analyst demo.

A portfolio of thirty agreements for a single fictional company — Meridian
Analytics Inc. (the "Customer" / in-house legal team) — spanning the document
types an SMB or enterprise legal team actually juggles: SaaS subscriptions,
NDAs, employment and contractor agreements, a separation/release, office and
equipment leases, master consulting/services agreements and a SOW, a master
services agreement, vendor supply and marketing agreements, a standalone DPA, a
perpetual software license, a reseller agreement, an evaluation agreement, a
master purchase agreement, and three amendments.

The corpus is engineered so the demo's hero queries — cross-document
reasoning that plain chunk-and-retrieve RAG cannot do — land hard. The value
comes from *planted tensions* across documents, catalogued below.

Core SaaS set (asserted field-by-field in test_queries.py — do not edit)
  saas-001  CloudStore Pro / Acme Corp — NY law, expires 2025-06-30, very low
            liability cap ($50 K) vs contract value ($500 K), GDPR + breach
            notification, net-30 payment.
  saas-002  DataSync Enterprise / TechVault Solutions — CA law, expires
            2026-06-30, cap = annual value ($240 K), audit rights, monthly /
            net-15 payment.
  saas-003  SecureVault Platform / Nexus Security Ltd — DE law, expires
            2025-12-31, high cap ($2 M), breach notification + GDPR + audit.
  saas-004  AnalyticsPro Suite / Acme Corp — NY law, expires 2027-03-31, low
            cap ($250 K) vs high value ($1.2 M), upfront payment (contradicts
            saas-001 net-30 for the same vendor), 90-day auto-renewal.
  saas-005  WorkflowManager Pro / Apex Systems — TX law, expires 2025-09-30,
            unusually long 180-day termination notice, competitor-assignment
            restriction.
  saas-001-amendment  Amendment No. 1 to saas-001 — term → 2027-06-30, cap →
            $500 K, net-45, subprocessor notice, fee → $540 K.

Extended portfolio (new)
  saas-006  Cobalt CRM / Cobalt Software — normal baseline: net-30, $180 K
            value, $180 K cap, 30-day notice, expires 2027-08-31 (a
            deliberately "unremarkable" contract so the outliers stand out).
  nda-001   Mutual NDA / Helix Biosciences — CA, expires 2026-08-31.
  nda-002   One-way NDA (Meridian discloses) / Brightpath Consulting — NY,
            expires 2025-11-30.
  nda-003   Mutual NDA / Northstar Analytics — GA, expires 2026-12-31.
  emp-001   Employment Agreement / Jordan Ellis (VP Engineering) — MA, at-will,
            12-month non-compete + non-solicit, no fixed expiry.
  emp-002   Executive Employment Agreement / Dana Frost (CFO) — DE, 24-month
            non-compete, change-of-control severance.
  emp-003   Independent Contractor Agreement / Priya Nair — WA, IP assignment,
            12-month non-solicit, NO non-compete, expires 2026-05-31.
  sep-001   Separation & Release Agreement / Alex Rivera — CA, reaffirms
            12-month non-solicit, release of claims.
  lease-001 Commercial Office Lease / Fifth Avenue Holdings — NY, expires
            2029-05-31, $780 K annual rent, 5-yr auto-renewal (12-month notice).
  lease-001-amendment  Amendment No. 1 to lease-001 — reduces premises & rent
            ($780 K → $560 K), extends term to 2031-05-31, notice → 9 months.
  lease-002 Equipment Lease / DataCore Leasing — TX, expires 2026-02-28,
            $95 K, early-termination penalty.
  msc-001   Master Consulting Services Agreement / Brightpath Consulting — NY,
            net-45, expires 2027-01-31, uncapped for confidentiality breach.
  sow-001   Statement of Work No. 1 under msc-001 / Brightpath — $220 K,
            milestone-based payment.
  svc-001   Professional Services Agreement / TechVault Solutions — CA, net-60
            (contradicts saas-002 net-15 for the same vendor), expires
            2026-10-31.
  svc-002   Managed IT Services Agreement / Sentinel IT — CO, expires
            2026-09-30, auto-renewal (90-day notice), SLA credits.
  msa-001   Master Services Agreement / Orion Logistics — IL, expires
            2028-06-30, UNCAPPED / unlimited liability (risk outlier).
  msa-001-amendment  Amendment No. 1 to msa-001 — adds a $1 M liability cap
            where none existed.
  vnd-001   Vendor Supply Agreement / Pinnacle Hardware — WA, expires
            2025-10-31, net-30, auto-renewal, NO data-protection/DPA clause.
  vnd-002   Marketing Services Agreement / Vivid Media Group — FL, expires
            2026-04-30, monthly, NO stated liability cap.
  dpa-001   Data Processing Addendum to saas-001 / Acme Corp — GDPR, SCCs,
            subprocessor list.
  lic-001   Perpetual Software License / Quantum Software Labs — MA, perpetual
            (no expiry), $150 K annual maintenance.
  rsl-001   Reseller / Channel Partner Agreement / Northstar Analytics — GA,
            expires 2027-09-30, exclusive territory, 20% margin, auto-renewal.
  grn-001   Software Evaluation Agreement / Beacon AI — CA, expires 2025-08-31,
            90-day trial that auto-converts to paid unless notice.
  pur-001   Master Purchase Agreement / Ironclad Manufacturing — OH, expires
            2027-05-31, net-30.

Planted cross-document signals (what the hero queries exploit)
  Same-counterparty payment conflict
    Acme:      saas-001 (net-30) vs saas-004 (upfront)
    TechVault: saas-002 (net-15 / monthly) vs svc-001 (net-60)
  Amendment flips a clause
    saas-001-amendment (term / cap / payment); lease-001-amendment (premises /
    rent / term); msa-001-amendment (adds a cap to a previously uncapped MSA)
  Liability outliers
    saas-003 $2 M cap (high); msa-001 UNCAPPED (extreme, until amended);
    vnd-002 & vnd-001 no cap stated (missing); low cap vs value: saas-001
    ($50 K/$500 K), saas-004 ($250 K/$1.2 M)
  Notice-period outlier      saas-005 (180 days) vs 30–90 elsewhere
  Auto-renewal cluster       saas-004, lease-001, svc-002, vnd-001, rsl-001,
                             grn-001
  Expiring before 2026-04    saas-001, saas-003, saas-005, nda-002, lease-002,
                             vnd-001, grn-001
  Non-compete / restrictive covenants
    emp-001 (12 mo), emp-002 (24 mo), emp-003 (non-solicit only), sep-001
    (reaffirms non-solicit), saas-005 (competitor-assignment restriction)
  Missing-clause risk        vnd-001 (no DPA), vnd-002 (no liability cap)
  GDPR / data residency      saas-001, saas-003, saas-004, dpa-001
  Governing-law spread       NY, CA, DE, TX, MA, IL, WA, FL, CO, GA, OH
  Parent/child references    sow-001 → msc-001; dpa-001 → saas-001;
                             the three amendments → their originals
"""

SAAS_001 = """\
SOFTWARE AS A SERVICE AGREEMENT

Agreement No.: MSA-2024-0391
Effective Date: April 1, 2024
Expiry Date: June 30, 2025

PARTIES

This Software as a Service Agreement ("Agreement") is entered into as of April 1, 2024,
between Meridian Analytics Inc., a Delaware corporation with its principal place of
business at 350 Fifth Avenue, New York, NY 10118 ("Customer"), and Acme Corp, a New York
corporation with its principal place of business at 100 Technology Drive, New York, NY 10001
("Provider").

PURPOSE

Provider agrees to furnish Customer with a cloud-hosted file-storage and collaboration
platform known as CloudStore Pro, enabling Customer's employees to store, share, and
co-edit documents in accordance with the terms herein.

1. SERVICES AND SUBSCRIPTION

Provider grants Customer a non-exclusive, non-transferable subscription to access CloudStore
Pro for up to 500 named users.  The annual subscription fee is USD 500,000, payable in
accordance with Section 4.

2. TERM

This Agreement commences on April 1, 2024 and expires on June 30, 2025 unless earlier
terminated pursuant to Section 8.

3. SERVICE LEVELS

Provider shall maintain 99.5% monthly uptime.  Scheduled maintenance windows shall not
exceed four hours per month and shall occur between 02:00 and 06:00 Eastern Time on
Sundays.

4. PAYMENT TERMS

Invoices are issued on the first business day of each calendar quarter.  Customer shall
remit payment within thirty (30) days of the invoice date (net-30).  Amounts outstanding
beyond thirty days accrue interest at 1.5% per month.  All amounts are in United States
Dollars and are exclusive of applicable taxes.

5. DATA PROTECTION AND GDPR

5.1  Provider processes personal data on behalf of Customer solely in accordance with
Customer's documented instructions.  Provider shall not process personal data for any
purpose other than fulfilling its obligations under this Agreement.

5.2  Provider shall implement and maintain appropriate technical and organisational measures
to protect personal data against unauthorised or unlawful processing, accidental loss,
destruction, or damage, including but not limited to AES-256 encryption at rest and TLS 1.3
in transit.

5.3  Data residency: All personal data of European Union data subjects shall be stored and
processed exclusively within data centres located in the European Economic Area.  Provider
shall not transfer such data to countries outside the EEA without Customer's prior written
consent and the implementation of appropriate transfer mechanisms under Chapter V of
Regulation (EU) 2016/679 ("GDPR").

5.4  Breach Notification: Provider shall notify Customer without undue delay and in any
event no later than 48 hours after becoming aware of a personal data breach affecting
Customer's data.  Such notification shall include: (a) a description of the nature of the
breach; (b) the categories and approximate number of data subjects concerned; (c) the
categories and approximate number of records concerned; (d) the likely consequences of the
breach; and (e) the measures taken or proposed to address the breach.

6. CONFIDENTIALITY

Each party agrees to hold the other party's Confidential Information in strict confidence
and not to disclose it to any third party without the prior written consent of the
disclosing party.  This obligation survives termination of the Agreement for a period of
five (5) years.  "Confidential Information" means any non-public information disclosed by
one party to the other, whether orally, in writing, or by other means.

7. LIMITATION OF LIABILITY

7.1  IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL,
SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS, REGARDLESS OF THE
FORM OF ACTION AND EVEN IF SUCH PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

7.2  EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL
NOT EXCEED USD 50,000, REPRESENTING THE LIABILITY CAP.  THE PARTIES ACKNOWLEDGE THAT THIS
LIMITATION REFLECTS A REASONABLE ALLOCATION OF RISK.

8. TERMINATION

8.1  Either party may terminate this Agreement for convenience upon thirty (30) days' prior
written notice to the other party.

8.2  Either party may terminate immediately upon written notice if the other party: (a)
materially breaches any provision of this Agreement and fails to cure such breach within
thirty (30) days of receiving written notice thereof; or (b) becomes insolvent, makes a
general assignment for the benefit of creditors, or files a voluntary petition in bankruptcy.

8.3  Upon termination Customer shall have thirty (30) days to export its data, after which
Provider may delete all Customer data.

9. INDEMNIFICATION

Each party ("Indemnifying Party") shall indemnify, defend, and hold harmless the other
party, its officers, directors, employees, and agents from and against any claims, damages,
costs, and expenses (including reasonable attorneys' fees) arising from: (a) the
Indemnifying Party's breach of any representation or warranty; (b) the Indemnifying
Party's gross negligence or wilful misconduct; or (c) in the case of Provider, any claim
that the Services infringe any third-party intellectual property right.

10. GOVERNING LAW AND DISPUTE RESOLUTION

This Agreement shall be governed by and construed in accordance with the laws of the State
of New York, without regard to its conflict-of-law principles.  Any dispute, controversy,
or claim arising out of or relating to this Agreement shall be submitted to binding
arbitration administered by the American Arbitration Association under its Commercial
Arbitration Rules.  The seat of arbitration shall be New York, New York.  The award shall
be final and binding and may be enforced in any court of competent jurisdiction.

11. MISCELLANEOUS

This Agreement constitutes the entire agreement between the parties with respect to its
subject matter and supersedes all prior agreements.  Any amendment must be in writing and
signed by authorised representatives of both parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written
above.

Meridian Analytics Inc.                Acme Corp
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Michael Torres
Title: Chief Procurement Officer       Title: Vice President, Enterprise Sales
"""

SAAS_002 = """\
ENTERPRISE SOFTWARE SUBSCRIPTION AGREEMENT

Reference: ESSA-2023-0774
Effective Date: July 1, 2023
Expiry Date: June 30, 2026

PARTIES

This Enterprise Software Subscription Agreement ("Agreement") is made between Meridian
Analytics Inc., a Delaware corporation ("Customer"), and TechVault Solutions, LLC, a
California limited liability company with its principal place of business at 2250 Market
Street, San Francisco, CA 94114 ("Provider").

PURPOSE

Provider shall furnish Customer with access to DataSync Enterprise, a cloud-based data
integration and pipeline management platform, enabling Customer to build, monitor, and
operate data pipelines across its enterprise systems.

1. SUBSCRIPTION SCOPE

Subject to the terms herein, Provider grants Customer a non-exclusive subscription for an
unlimited number of users within Customer's organisation.  Annual subscription fee: USD
240,000, billed monthly at USD 20,000.

2. TERM

This Agreement commences on July 1, 2023 and expires on June 30, 2026.

3. PAYMENT TERMS

Provider shall issue monthly invoices on the first day of each calendar month.  Customer
shall pay each invoice within fifteen (15) days of receipt.  Late payments shall accrue
interest at the lower of 1% per month or the maximum rate permitted by applicable law.
Payment shall be made by ACH transfer to the bank account designated by Provider.  All
fees are non-refundable except as expressly set forth herein.

4. AUDIT RIGHTS

Customer shall have the right, upon thirty (30) days' prior written notice and no more than
once per calendar year, to audit Provider's technical and organisational security controls,
data handling procedures, and compliance with the terms of this Agreement.  Such audit
shall be conducted during normal business hours by Customer or a mutually agreed third-party
auditor at Customer's expense.  Provider shall make relevant records, systems, and personnel
reasonably available for the audit.  Customer shall treat all information obtained during
such audit as Provider's Confidential Information.

5. CONFIDENTIALITY

Each party shall protect the other's Confidential Information using at least the same degree
of care it uses to protect its own confidential information, but in no event less than
reasonable care.  This obligation survives for three (3) years after termination.

6. LIMITATION OF LIABILITY

PROVIDER'S AGGREGATE LIABILITY UNDER THIS AGREEMENT FOR ANY AND ALL CLAIMS SHALL NOT
EXCEED THE TOTAL FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS IMMEDIATELY PRECEDING
THE CLAIM (NOT TO EXCEED USD 240,000).  NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

7. TERMINATION

7.1  Either party may terminate this Agreement for convenience by providing sixty (60) days'
prior written notice.

7.2  Provider may suspend access (rather than terminate) upon Customer's failure to pay
any amount due within thirty (30) days after the due date, and may terminate if such
failure continues for an additional thirty (30) days after notice of suspension.

7.3  Following termination, Provider shall provide Customer with a data export in CSV or
JSON format within fifteen (15) business days upon request.

8. INDEMNIFICATION

Provider shall indemnify and hold Customer harmless from any third-party intellectual
property infringement claims directly arising from Customer's authorised use of the
Services.  Customer shall indemnify Provider against claims arising from Customer's data
or Customer's use of the Services in violation of applicable law.

9. GOVERNING LAW

This Agreement is governed by the laws of the State of California, excluding its conflict-
of-law rules.  Any litigation arising hereunder shall be brought exclusively in the state
or federal courts located in San Francisco County, California, and the parties consent to
personal jurisdiction therein.

10. ENTIRE AGREEMENT

This Agreement and its exhibits represent the complete agreement between the parties on the
subject matter herein.

Meridian Analytics Inc.                TechVault Solutions, LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Priya Mehta
Title: Chief Procurement Officer       Title: General Counsel
"""

SAAS_003 = """\
CLOUD SECURITY PLATFORM SUBSCRIPTION AGREEMENT

Contract ID: CSPSA-2024-0512
Effective Date: January 1, 2024
Expiry Date: December 31, 2025

PARTIES

This Cloud Security Platform Subscription Agreement ("Agreement") is entered into between
Meridian Analytics Inc., a Delaware corporation ("Customer"), and Nexus Security Ltd., a
company incorporated in Delaware with principal offices at 890 Innovation Way, Wilmington,
DE 19801 ("Provider").

PURPOSE

Provider shall provide Customer with access to SecureVault Platform, a cloud-native security
operations platform offering continuous vulnerability scanning, threat detection, incident
response workflow automation, and compliance reporting.

1. SUBSCRIPTION AND FEES

Provider grants Customer a subscription for up to 1,000 managed endpoints.  The total
annual fee is USD 180,000, paid upfront within fifteen (15) days of the Effective Date.
Additional endpoint blocks (in increments of 100) are available at USD 15,000 per block
per year.

2. TERM

This Agreement runs from January 1, 2024 to December 31, 2025.

3. PERSONAL DATA, GDPR, AND DATA RESIDENCY

3.1  To the extent Provider processes personal data on behalf of Customer, Provider acts as
a data processor within the meaning of the GDPR and agrees to be bound by the Data
Processing Addendum attached as Exhibit A, which is incorporated herein by reference.

3.2  All Customer data, including personal data of EU data subjects, shall be stored
exclusively in ISO 27001-certified data centres within the European Union.  Provider shall
not sub-process or transfer Customer data outside the EU without Customer's prior written
consent and execution of EU Standard Contractual Clauses.

3.3  Breach Notification: Provider shall notify Customer by telephone and email within
24 hours of discovering any actual or reasonably suspected unauthorised access to,
disclosure of, or use of Customer data.  Provider shall follow up with a written incident
report within 72 hours setting out: (a) the scope and nature of the incident; (b) data
categories and volume affected; (c) probable cause; (d) immediate containment measures
taken; and (e) proposed remediation timeline.

4. AUDIT RIGHTS

Customer may, upon fourteen (14) days' written notice, conduct or commission a security
audit of Provider's infrastructure and processes, including penetration testing, no more
than twice per calendar year.  Provider shall cooperate fully and grant auditors read-only
access to relevant logging and configuration systems.  Audit reports and findings are
Customer's Confidential Information.  Provider shall remediate material findings within
sixty (60) days of written notification.

5. CONFIDENTIALITY

Each party shall maintain the confidentiality of the other's Confidential Information for
the duration of this Agreement and for five (5) years thereafter.

6. LIMITATION OF LIABILITY

6.1  PROVIDER'S TOTAL AGGREGATE LIABILITY FOR ALL CLAIMS ARISING UNDER THIS AGREEMENT,
WHETHER IN CONTRACT, TORT, OR OTHERWISE, SHALL NOT EXCEED USD 2,000,000.

6.2  THE LIABILITY CAP IN SECTION 6.1 SHALL NOT APPLY TO: (A) BREACHES OF
CONFIDENTIALITY; (B) PROVIDER'S INDEMNIFICATION OBLIGATIONS; OR (C) CLAIMS ARISING FROM
PROVIDER'S GROSS NEGLIGENCE OR WILFUL MISCONDUCT.

6.3  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL
DAMAGES.

7. TERMINATION

7.1  Either party may terminate for convenience upon ninety (90) days' prior written notice.

7.2  Customer may terminate immediately if Provider suffers a material security breach and
fails to provide a remediation plan acceptable to Customer within fifteen (15) days.

7.3  Provider may terminate immediately upon Customer's material breach of the data
protection provisions in Section 3 that is not cured within ten (10) days of notice.

8. INDEMNIFICATION

Provider shall indemnify Customer against third-party claims arising from: (a) Provider's
breach of its data protection obligations; (b) Provider's infringement of third-party
intellectual property; and (c) personal data breaches attributable to Provider's systems.
Customer's aggregate indemnification liability shall not exceed the total fees paid in the
preceding twelve (12) months.

9. GOVERNING LAW

This Agreement shall be governed by the laws of the State of Delaware.  The parties agree
that any disputes shall first be referred to senior management for informal resolution.
If not resolved within thirty (30) days, disputes shall be submitted to binding arbitration
under JAMS Streamlined Arbitration Rules, with the seat in Wilmington, Delaware.

Meridian Analytics Inc.                Nexus Security Ltd.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Dr. Elise Wagner
Title: Chief Procurement Officer       Title: Chief Executive Officer
"""

SAAS_004 = """\
ANALYTICS PLATFORM SUBSCRIPTION AGREEMENT

Agreement Reference: APSA-2024-0887
Effective Date: October 1, 2024
Expiry Date: March 31, 2027

PARTIES

This Analytics Platform Subscription Agreement ("Agreement") is made as of October 1, 2024,
between Meridian Analytics Inc., a Delaware corporation with its principal place of business
at 350 Fifth Avenue, New York, NY 10118 ("Customer"), and Acme Corp, a New York corporation
with its principal place of business at 100 Technology Drive, New York, NY 10001
("Provider").

PURPOSE

Provider shall furnish Customer with a subscription to AnalyticsPro Suite, an enterprise
business intelligence and analytics platform enabling large-scale data visualisation,
predictive modelling, and automated reporting across Customer's business units.

1. SUBSCRIPTION AND FEES

Provider grants Customer an enterprise-wide subscription with unlimited named users.
The total annual subscription fee is USD 1,200,000.  The full annual amount for each
subscription year is due and payable upfront within ten (10) business days of the start
of each contract year.  No refunds are available for early termination initiated by
Customer.

2. TERM AND AUTO-RENEWAL

2.1  This Agreement commences October 1, 2024 and expires March 31, 2027, unless earlier
terminated.

2.2  Auto-Renewal: Unless either party provides written notice of non-renewal at least
ninety (90) days before the then-current expiry date, this Agreement shall automatically
renew for successive one-year periods on the same terms and conditions, subject to a fee
increase of no more than 8% per renewal year.

3. DATA PROTECTION AND GDPR

3.1  Provider shall process Customer's personal data solely as a data processor under the
GDPR.  Provider agrees to maintain appropriate technical and organisational safeguards,
including SOC 2 Type II certification, renewed annually.

3.2  Data Residency: Personal data relating to EU and UK data subjects shall be stored
exclusively in Provider's EU-region infrastructure.  Provider shall not transfer such data
outside the EEA or UK without execution of applicable Standard Contractual Clauses and
Customer's prior written consent.

3.3  Provider shall maintain a record of processing activities relating to Customer data
and shall make such record available to Customer upon written request within five (5)
business days.

4. CONFIDENTIALITY

Each party agrees to protect the other's Confidential Information with the same degree of
care used to protect its own proprietary information, but no less than reasonable care.
This obligation survives five (5) years beyond termination.

5. LIMITATION OF LIABILITY

5.1  IN NO EVENT SHALL PROVIDER'S AGGREGATE LIABILITY EXCEED USD 250,000 (THE "LIABILITY
CAP"), REGARDLESS OF THE FORM OF ACTION OR THE BASIS OF THE CLAIM, INCLUDING CONTRACT,
TORT, OR STATUTE.

5.2  THE LIABILITY CAP SHALL NOT APPLY IN CASES OF: (A) PROVIDER'S FRAUD OR WILFUL
MISCONDUCT; OR (B) PROVIDER'S INDEMNIFICATION OBLIGATIONS UNDER SECTION 9 FOR THIRD-PARTY
INTELLECTUAL PROPERTY CLAIMS.

5.3  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES.

6. TERMINATION

6.1  Either party may terminate for convenience upon thirty (30) days' prior written notice.
Customer termination for convenience shall not entitle Customer to any refund of prepaid
fees.

6.2  Either party may terminate for cause if the other party materially breaches this
Agreement and fails to cure such breach within thirty (30) days of written notice.

6.3  Provider may terminate with immediate effect if Customer's use of the Services
materially violates applicable data protection law.

7. INDEMNIFICATION

9.1  Provider shall indemnify and hold Customer harmless from third-party claims alleging
that the Services infringe any copyright, patent, or trade secret of a third party.
Provider's indemnification obligations are uncapped.

9.2  Customer shall indemnify Provider against claims arising from Customer's violation of
applicable law or misuse of the Services.

8. GOVERNING LAW AND DISPUTE RESOLUTION

This Agreement is governed by the laws of the State of New York.  The parties shall
first attempt to resolve any dispute through good-faith negotiation between senior
representatives.  If not resolved within forty-five (45) days, either party may commence
binding arbitration before the American Arbitration Association under its Commercial
Arbitration Rules, seated in New York, New York.

Meridian Analytics Inc.                Acme Corp
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Michael Torres
Title: Chief Procurement Officer       Title: Vice President, Enterprise Sales
"""

SAAS_005 = """\
ENTERPRISE WORKFLOW MANAGEMENT SUBSCRIPTION AGREEMENT

Contract Reference: EWMSA-2023-0312
Effective Date: October 1, 2023
Expiry Date: September 30, 2025

PARTIES

This Enterprise Workflow Management Subscription Agreement ("Agreement") is entered into
as of October 1, 2023, between Meridian Analytics Inc., a Delaware corporation
("Customer"), and Apex Systems Inc., a Texas corporation with its principal place of
business at 4400 Post Oak Parkway, Houston, TX 77027 ("Provider").

PURPOSE

Provider shall provide Customer with a subscription to WorkflowManager Pro, a cloud-based
business process automation and workflow management platform, enabling Customer to design,
deploy, and monitor enterprise workflows across its operations.

1. SUBSCRIPTION AND FEES

Provider grants Customer a departmental subscription covering up to 200 named users across
Customer's Operations and Finance departments.  Annual subscription fee: USD 360,000.
The fee for the initial twelve (12) months is payable within thirty (30) days of the
Effective Date.  Subsequent annual fees are due on each anniversary of the Effective Date.
Payments overdue by more than fifteen (15) days shall accrue a late fee of 2% per month on
the outstanding balance.

2. TERM

This Agreement runs from October 1, 2023 through September 30, 2025.

3. CONFIDENTIALITY

Each party shall hold the other party's Confidential Information in strict confidence and
shall not disclose it to any third party except: (a) with the prior written consent of the
disclosing party; (b) to employees or contractors who have a need to know and are bound by
confidentiality obligations at least as protective as those herein; or (c) as required by
applicable law or court order.  This obligation continues for four (4) years after
termination.

4. ASSIGNMENT AND CHANGE OF CONTROL

4.1  Neither party may assign this Agreement or any rights or obligations hereunder without
the prior written consent of the other party, which shall not be unreasonably withheld.

4.2  Notwithstanding Section 4.1, Customer expressly prohibits Provider from assigning this
Agreement, or subcontracting any material portion of the Services, to a Competitor of
Customer.  For purposes of this Section, "Competitor" means any entity that derives more
than 20% of its annual revenue from business process automation or workflow management
products that directly compete with Customer's core offerings.  Any purported assignment
to a Competitor is void ab initio.

4.3  A change of control of Provider (including by merger, acquisition, or sale of all or
substantially all of Provider's assets) to a Competitor shall constitute a material breach
of this Agreement entitling Customer to terminate immediately without penalty.

5. LIMITATION OF LIABILITY

PROVIDER'S AGGREGATE LIABILITY UNDER OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED USD
500,000.  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, CONSEQUENTIAL, SPECIAL, OR
PUNITIVE DAMAGES, REGARDLESS OF THE FORM OF ACTION.  THE FOREGOING LIMITATIONS SHALL NOT
APPLY TO EITHER PARTY'S INDEMNIFICATION OBLIGATIONS OR TO LIABILITY ARISING FROM FRAUD
OR WILFUL MISCONDUCT.

6. TERMINATION

6.1  Either party may terminate this Agreement for convenience upon one hundred eighty
(180) days' prior written notice to the other party.  This extended notice period reflects
the operational dependency Customer has on the platform and the time required for Customer
to complete a migration to an alternative solution.

6.2  Either party may terminate for cause upon written notice if the other party: (a)
materially breaches any provision of this Agreement and fails to cure such breach within
thirty (30) days; or (b) ceases to conduct business in the ordinary course.

6.3  Provider may suspend Services immediately if Customer's account is more than sixty
(60) days past due, and may terminate if such arrearage is not cured within fifteen (15)
additional days following written notice.

7. INDEMNIFICATION

Provider shall defend, indemnify, and hold harmless Customer from third-party claims
arising from: (a) Provider's breach of this Agreement; (b) Provider's negligence or wilful
misconduct; or (c) infringement of third-party intellectual property by the Services.
Customer shall indemnify Provider from claims arising from Customer's breach or misuse of
the Services.

8. GOVERNING LAW AND DISPUTE RESOLUTION

This Agreement shall be governed by the laws of the State of Texas, without regard to
conflicts-of-law principles.  The parties agree that any dispute shall be resolved by
binding arbitration under the rules of the American Arbitration Association, with the seat
of arbitration in Houston, Texas.  In addition to its claim or counterclaim in arbitration,
either party may seek emergency injunctive relief in any court of competent jurisdiction to
prevent irreparable harm pending the conclusion of arbitration.

9. ENTIRE AGREEMENT

This Agreement supersedes all prior negotiations, representations, and agreements relating
to its subject matter.

Meridian Analytics Inc.                Apex Systems Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Robert Kimball
Title: Chief Procurement Officer       Title: President & CEO
"""

SAAS_001_AMENDMENT = """\
AMENDMENT NO. 1 TO SOFTWARE AS A SERVICE AGREEMENT

Amendment No.: MSA-2024-0391-A1
Effective Date of Amendment: July 1, 2025

PARTIES

This Amendment No. 1 (this "Amendment") is entered into as of July 1, 2025, between
Meridian Analytics Inc., a Delaware corporation with its principal place of business at
350 Fifth Avenue, New York, NY 10118 ("Customer"), and Acme Corp, a New York corporation
with its principal place of business at 100 Technology Drive, New York, NY 10001
("Provider").

RECITALS

WHEREAS, the parties entered into that certain Software as a Service Agreement No.
MSA-2024-0391, effective April 1, 2024 (the "Agreement"), pursuant to which Provider
furnishes Customer the cloud-hosted platform known as CloudStore Pro; and

WHEREAS, the parties wish to extend the term, increase the limitation of liability cap,
and revise the payment terms of the Agreement;

NOW, THEREFORE, in consideration of the mutual covenants set forth herein, the parties
agree as follows:

1. TERM EXTENSION

Section 2 (Term) of the Agreement is deleted in its entirety and replaced with the
following: "This Agreement commences on April 1, 2024 and expires on June 30, 2027 unless
earlier terminated pursuant to Section 8."

2. LIMITATION OF LIABILITY

Section 7.2 of the Agreement is deleted in its entirety and replaced with the following:
"EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT
EXCEED USD 500,000, REPRESENTING THE LIABILITY CAP. THE PARTIES ACKNOWLEDGE THAT THIS
LIMITATION REFLECTS A REASONABLE ALLOCATION OF RISK."

3. PAYMENT TERMS

The second sentence of Section 4 (Payment Terms) of the Agreement is deleted and replaced
with the following: "Customer shall remit payment within forty-five (45) days of the
invoice date (net-45)." All other provisions of Section 4 remain unchanged.

4. NEW SECTION 5.5 – SUBPROCESSOR NOTICE

A new Section 5.5 is added to the Agreement immediately following Section 5.4, reading in
its entirety: "5.5 Subprocessors. Provider shall maintain an up-to-date list of
subprocessors engaged in the processing of Customer personal data and shall provide
Customer no less than thirty (30) days' prior written notice before adding or replacing
any subprocessor, during which period Customer may object on reasonable data-protection
grounds."

5. ANNUAL SUBSCRIPTION FEE

The annual subscription fee set forth in Section 1 is increased from USD 500,000 to
USD 540,000, effective for each contract year beginning on or after July 1, 2025,
reflecting the extended term and expanded data-protection commitments.

6. NO OTHER CHANGES

Except as expressly amended herein, all terms and conditions of the Agreement remain in
full force and effect. In the event of any conflict between this Amendment and the
Agreement, this Amendment shall control. Capitalized terms not defined herein have the
meanings given to them in the Agreement.

IN WITNESS WHEREOF, the parties have executed this Amendment as of the date first written
above.

Meridian Analytics Inc.                Acme Corp
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Michael Torres
Title: Chief Procurement Officer       Title: Vice President, Enterprise Sales
"""


SAAS_006 = """\
SOFTWARE AS A SERVICE AGREEMENT

Agreement No.: CRM-2024-1180
Effective Date: September 1, 2024
Expiry Date: August 31, 2027

PARTIES

This Software as a Service Agreement ("Agreement") is entered into as of September 1, 2024,
between Meridian Analytics Inc., a Delaware corporation with its principal place of business
at 350 Fifth Avenue, New York, NY 10118 ("Customer"), and Cobalt Software Inc., a Washington
corporation with its principal place of business at 1201 Third Avenue, Seattle, WA 98101
("Provider").

PURPOSE

Provider shall provide Customer with access to Cobalt CRM, a cloud-hosted customer
relationship management platform for Customer's sales and account-management teams.

1. SUBSCRIPTION AND FEES

Provider grants Customer a non-exclusive subscription for up to 300 named users.  The
annual subscription fee is USD 180,000.

2. TERM

This Agreement commences September 1, 2024 and expires August 31, 2027 unless earlier
terminated pursuant to Section 6.

3. PAYMENT TERMS

Provider shall issue invoices annually in advance of each contract year.  Customer shall
remit payment within thirty (30) days of the invoice date (net-30).  Overdue amounts accrue
interest at 1% per month.

4. SERVICE LEVELS

Provider shall maintain 99.9% monthly uptime measured on a calendar-month basis, excluding
scheduled maintenance not exceeding two hours per week.

5. LIMITATION OF LIABILITY

EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT
EXCEED USD 180,000.  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, OR
CONSEQUENTIAL DAMAGES.

6. TERMINATION

Either party may terminate for convenience upon thirty (30) days' prior written notice, or
for cause upon a material breach uncured within thirty (30) days of written notice.

7. CONFIDENTIALITY

Each party shall protect the other's Confidential Information with reasonable care for the
term of this Agreement and three (3) years thereafter.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of Washington.  The parties consent to
the exclusive jurisdiction of the state and federal courts located in King County,
Washington.

Meridian Analytics Inc.                Cobalt Software Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Grace Lin
Title: Chief Procurement Officer       Title: VP Sales
"""

NDA_001 = """\
MUTUAL NON-DISCLOSURE AGREEMENT

Reference: MNDA-2023-0455
Effective Date: September 1, 2023

PARTIES

This Mutual Non-Disclosure Agreement ("Agreement") is entered into between Meridian
Analytics Inc., a Delaware corporation ("Meridian"), and Helix Biosciences Inc., a
California corporation with its principal place of business at 500 Forbes Boulevard, South
San Francisco, CA 94080 ("Helix").  Each party may act as both a Disclosing Party and a
Receiving Party.

1. PURPOSE

The parties wish to explore a potential collaboration relating to analytics tooling for
clinical-trial data (the "Purpose") and, in connection therewith, may disclose Confidential
Information to each other.

2. CONFIDENTIAL INFORMATION

"Confidential Information" means any non-public business, technical, financial, or product
information disclosed by one party to the other, whether orally, in writing, or by
inspection, that is marked confidential or that a reasonable person would understand to be
confidential given its nature and the circumstances of disclosure.

3. OBLIGATIONS

The Receiving Party shall: (a) use the Confidential Information solely for the Purpose; (b)
protect it with at least the same degree of care it uses for its own confidential
information of like importance, and no less than reasonable care; and (c) not disclose it to
any third party except to its employees, affiliates, and advisors who have a need to know
and are bound by confidentiality obligations no less protective than those herein.

4. EXCLUSIONS

Confidential Information does not include information that: (a) is or becomes public through
no fault of the Receiving Party; (b) was rightfully known to the Receiving Party without
restriction before disclosure; (c) is independently developed without use of the Disclosing
Party's Confidential Information; or (d) is rightfully obtained from a third party without
restriction.

5. TERM

This Agreement commences on the Effective Date and expires on August 31, 2026.  The
confidentiality obligations survive for three (3) years following disclosure of the
Confidential Information.

6. NO LICENSE; NO OBLIGATION

No license or other right is granted under any intellectual property.  Nothing herein
obligates either party to proceed with any transaction.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of California, without regard to its
conflict-of-law principles.

Meridian Analytics Inc.                Helix Biosciences Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Dr. Omar Haddad
Title: Chief Procurement Officer       Title: Chief Business Officer
"""

NDA_002 = """\
NON-DISCLOSURE AGREEMENT (ONE-WAY)

Reference: NDA-2024-0733
Effective Date: December 1, 2024

PARTIES

This Non-Disclosure Agreement ("Agreement") is entered into between Meridian Analytics
Inc., a Delaware corporation ("Discloser"), and Brightpath Consulting LLC, a New York
limited liability company with its principal place of business at 1 Bryant Park, New York,
NY 10036 ("Recipient").

1. PURPOSE

Discloser wishes to disclose certain Confidential Information to Recipient so that Recipient
may evaluate and scope a potential consulting engagement relating to Discloser's data
platform (the "Purpose").

2. CONFIDENTIAL INFORMATION

"Confidential Information" means all non-public information disclosed by Discloser to
Recipient, including product roadmaps, architecture, customer lists, pricing, and financial
data, whether or not marked confidential.

3. OBLIGATIONS OF RECIPIENT

Recipient shall use the Confidential Information solely for the Purpose, shall not disclose
it to any third party, and shall restrict access to those of its personnel who need it for
the Purpose and who are bound by written confidentiality obligations.  Recipient shall
return or destroy all Confidential Information upon Discloser's written request.

4. TERM

This Agreement is effective from the Effective Date and expires on November 30, 2025.
Recipient's confidentiality obligations survive for a period of five (5) years from the date
of disclosure.

5. REMEDIES

Recipient acknowledges that a breach of this Agreement may cause irreparable harm for which
monetary damages are inadequate, and that Discloser shall be entitled to seek injunctive
relief in addition to any other remedies available at law or in equity.

6. GOVERNING LAW

This Agreement is governed by the laws of the State of New York.  The parties submit to the
exclusive jurisdiction of the state and federal courts located in New York County, New York.

Meridian Analytics Inc.                Brightpath Consulting LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Laura Devine
Title: Chief Procurement Officer       Title: Managing Partner
"""

NDA_003 = """\
MUTUAL NON-DISCLOSURE AGREEMENT

Reference: MNDA-2024-0902
Effective Date: January 15, 2024

PARTIES

This Mutual Non-Disclosure Agreement ("Agreement") is between Meridian Analytics Inc., a
Delaware corporation ("Meridian"), and Northstar Analytics Inc., a Georgia corporation with
its principal place of business at 1180 Peachtree Street NE, Atlanta, GA 30309
("Northstar"), entered into in contemplation of a potential reseller relationship.

1. CONFIDENTIAL INFORMATION

Each party may disclose non-public commercial, technical, and financial information to the
other.  "Confidential Information" is any such information a reasonable person would treat
as confidential given its nature and the circumstances of disclosure.

2. OBLIGATIONS

The Receiving Party shall use Confidential Information solely to evaluate and pursue the
contemplated relationship, shall protect it with no less than reasonable care, and shall not
disclose it except to personnel and advisors with a need to know who are bound by comparable
obligations.

3. TERM

This Agreement expires on December 31, 2026.  Confidentiality obligations survive for three
(3) years after each disclosure.  Trade secrets remain protected for as long as they qualify
as trade secrets under applicable law.

4. GOVERNING LAW

This Agreement is governed by the laws of the State of Georgia.

Meridian Analytics Inc.                Northstar Analytics Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Marcus Webb
Title: Chief Procurement Officer       Title: Chief Revenue Officer
"""

EMP_001 = """\
EMPLOYMENT AGREEMENT

Reference: EMP-2024-0021
Effective Date: February 1, 2024

PARTIES

This Employment Agreement ("Agreement") is entered into between Meridian Analytics Inc., a
Delaware corporation with offices at 350 Fifth Avenue, New York, NY 10118 (the "Company"),
and Jordan Ellis, an individual residing in Cambridge, Massachusetts (the "Employee").

1. POSITION AND DUTIES

The Company employs Employee as Vice President, Engineering.  Employee shall devote full
business time and best efforts to the Company and shall report to the Chief Technology
Officer.

2. AT-WILL EMPLOYMENT

Employment is at-will.  Either the Company or Employee may terminate the employment
relationship at any time, with or without cause and with or without notice, subject to the
severance provisions of Section 6.

3. COMPENSATION

Employee's annual base salary is USD 265,000, payable in accordance with the Company's
standard payroll practices, plus eligibility for an annual target bonus of 25% and equity
awards under the Company's incentive plan.

4. CONFIDENTIALITY AND INVENTIONS

Employee shall hold the Company's Confidential Information in strict confidence during and
after employment.  Employee assigns to the Company all right, title, and interest in
inventions, works of authorship, and other intellectual property conceived or developed in
the course of employment.

5. NON-COMPETITION AND NON-SOLICITATION

5.1  During employment and for a period of twelve (12) months following the termination of
employment for any reason, Employee shall not, within any state in which the Company does
business, engage in or provide services to any business that competes with the Company's
data-analytics products.

5.2  During employment and for twelve (12) months thereafter, Employee shall not directly or
indirectly solicit for employment any employee of the Company or solicit any customer of the
Company with whom Employee had material contact.

5.3  The Company shall provide Employee with the Massachusetts-mandated garden-leave
consideration during the restricted period as required under M.G.L. c. 149, § 24L.

6. SEVERANCE

If the Company terminates Employee without cause, Employee shall receive three (3) months'
base salary as severance, conditioned on execution of a release of claims.

7. GOVERNING LAW

This Agreement is governed by the laws of the Commonwealth of Massachusetts.

Meridian Analytics Inc.                Employee
By: ____________________________       ____________________________
Name: Sarah Chen                       Jordan Ellis
Title: Chief People Officer
"""

EMP_002 = """\
EXECUTIVE EMPLOYMENT AGREEMENT

Reference: EMP-2023-0009
Effective Date: June 1, 2023

PARTIES

This Executive Employment Agreement ("Agreement") is between Meridian Analytics Inc., a
Delaware corporation (the "Company"), and Dana Frost, an individual (the "Executive").

1. POSITION

The Company employs Executive as Chief Financial Officer, reporting to the Chief Executive
Officer.

2. COMPENSATION

Executive's annual base salary is USD 340,000, with an annual target bonus of 40% and equity
participation as approved by the Board.

3. TERM AND TERMINATION

Employment is at-will.  The Company may terminate Executive with or without cause; Executive
may resign with or without good reason, in each case subject to Sections 5 and 6.

4. CONFIDENTIALITY AND INTELLECTUAL PROPERTY

Executive shall protect the Company's Confidential Information at all times and assigns to
the Company all work product and inventions arising from employment.

5. NON-COMPETITION AND NON-SOLICITATION

During employment and for twenty-four (24) months following termination, Executive shall
not (a) render services to any enterprise that competes with the Company in the data
analytics or business intelligence markets, or (b) solicit any employee, contractor, or
customer of the Company.  Executive acknowledges that this restriction is reasonable in
scope, geography, and duration given Executive's access to strategic and financial
information.

6. CHANGE OF CONTROL SEVERANCE

If, within twelve (12) months following a Change of Control, the Company terminates
Executive without cause or Executive resigns for good reason, Executive shall receive a
lump-sum payment equal to twelve (12) months' base salary plus target bonus, accelerated
vesting of all outstanding equity, and twelve (12) months of continued health benefits,
conditioned on a release of claims.  "Change of Control" means a merger, consolidation, or
sale of all or substantially all of the Company's assets following which the Company's
pre-transaction stockholders hold less than 50% of the voting power of the surviving entity.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of Delaware.

Meridian Analytics Inc.                Executive
By: ____________________________       ____________________________
Name: Sarah Chen                       Dana Frost
Title: Chief People Officer
"""

EMP_003 = """\
INDEPENDENT CONTRACTOR AGREEMENT

Reference: ICA-2024-0304
Effective Date: June 1, 2024
Expiry Date: May 31, 2026

PARTIES

This Independent Contractor Agreement ("Agreement") is entered into between Meridian
Analytics Inc., a Delaware corporation (the "Company"), and Priya Nair, an individual doing
business as a sole proprietor in Seattle, Washington (the "Contractor").

1. SERVICES

Contractor shall provide data-engineering and pipeline-development services as described in
one or more work orders issued under this Agreement.

2. INDEPENDENT CONTRACTOR STATUS

Contractor is an independent contractor and not an employee, partner, or agent of the
Company.  Contractor is responsible for all taxes on amounts paid hereunder and is not
eligible for Company employee benefits.

3. TERM

This Agreement runs from June 1, 2024 through May 31, 2026 unless earlier terminated on
thirty (30) days' written notice by either party.

4. COMPENSATION

The Company shall pay Contractor USD 180 per hour, invoiced monthly, payable net-30.

5. INTELLECTUAL PROPERTY

All deliverables, inventions, and works of authorship created by Contractor in performing
the Services are works made for hire and, to the extent they do not so qualify, are hereby
assigned to the Company.

6. NON-SOLICITATION

During the term and for twelve (12) months thereafter, Contractor shall not solicit for
employment or engagement any employee or contractor of the Company.  For the avoidance of
doubt, this Agreement contains no non-competition covenant, and Contractor remains free to
provide services to other clients, including competitors of the Company, provided Contractor
does not use or disclose the Company's Confidential Information.

7. CONFIDENTIALITY

Contractor shall keep the Company's Confidential Information confidential during and after
the term and shall not use it except to perform the Services.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of Washington.

Meridian Analytics Inc.                Contractor
By: ____________________________       ____________________________
Name: Sarah Chen                       Priya Nair
Title: Chief People Officer
"""

SEP_001 = """\
SEPARATION AND GENERAL RELEASE AGREEMENT

Reference: SEP-2024-0117
Effective Date: April 15, 2024

PARTIES

This Separation and General Release Agreement ("Agreement") is entered into between Meridian
Analytics Inc., a Delaware corporation (the "Company"), and Alex Rivera, a former employee
residing in Oakland, California ("Employee").

RECITALS

Employee's employment with the Company ended effective April 12, 2024.  The parties wish to
resolve all matters relating to Employee's employment and separation.

1. SEPARATION PAYMENT

In consideration for the promises herein, the Company shall pay Employee a separation payment
equal to four (4) months' base salary, less applicable withholdings, within thirty (30) days
of the Effective Date and after expiration of the revocation period in Section 5.

2. GENERAL RELEASE

Employee, on behalf of Employee and Employee's heirs and assigns, releases and forever
discharges the Company and its officers, directors, and employees from any and all claims,
known or unknown, arising out of or relating to Employee's employment or separation,
including claims under federal, state, and local employment laws, to the maximum extent
permitted by law.

3. CONTINUING OBLIGATIONS

Employee reaffirms the continuing confidentiality and intellectual-property assignment
obligations from Employee's prior Employment Agreement.  Employee further agrees that, for
twelve (12) months following the separation date, Employee shall not solicit any employee of
the Company to leave the Company's employ.  The parties acknowledge that, consistent with
California law, this Agreement contains no covenant restricting Employee's ability to compete
or to practice Employee's profession.

4. NON-DISPARAGEMENT

Each party agrees not to make disparaging statements about the other, subject to truthful
statements required by law or legal process.

5. ADEA / REVOCATION

Employee acknowledges having been given twenty-one (21) days to consider this Agreement and
seven (7) days to revoke it following execution, during which the Agreement is not
effective.

6. GOVERNING LAW

This Agreement is governed by the laws of the State of California.

Meridian Analytics Inc.                Employee
By: ____________________________       ____________________________
Name: Sarah Chen                       Alex Rivera
Title: Chief People Officer
"""

LEASE_001 = """\
COMMERCIAL OFFICE LEASE AGREEMENT

Reference: LEASE-2024-0055
Effective Date: June 1, 2024
Expiry Date: May 31, 2029

PARTIES

This Commercial Office Lease Agreement ("Lease") is entered into between Fifth Avenue
Holdings LLC, a New York limited liability company ("Landlord"), and Meridian Analytics
Inc., a Delaware corporation ("Tenant").

1. PREMISES

Landlord leases to Tenant approximately 22,000 rentable square feet on the 30th floor of the
building located at 350 Fifth Avenue, New York, NY 10118 (the "Premises").

2. TERM

The initial term is five (5) years, commencing June 1, 2024 and expiring May 31, 2029.

3. RENT

Base rent is USD 780,000 per annum, payable in equal monthly installments of USD 65,000 in
advance on the first day of each month.  Rent increases by three percent (3%) on each
anniversary of the commencement date.

4. RENEWAL

4.1  Provided Tenant is not in default, this Lease shall automatically renew for one
additional five (5)-year term at the then-prevailing fair market rent unless either party
delivers written notice of non-renewal at least twelve (12) months before the expiry date.

5. SECURITY DEPOSIT

Tenant shall deposit USD 130,000 as security for its obligations, refundable within sixty
(60) days after the term ends, less amounts properly applied by Landlord.

6. USE AND MAINTENANCE

The Premises shall be used solely for general office purposes.  Tenant shall maintain the
interior of the Premises; Landlord shall maintain the building structure, common areas, and
base building systems.

7. ASSIGNMENT AND SUBLETTING

Tenant shall not assign this Lease or sublet the Premises without Landlord's prior written
consent, which shall not be unreasonably withheld.

8. DEFAULT AND REMEDIES

If Tenant fails to pay rent within ten (10) days of its due date or breaches any other
material term uncured for thirty (30) days after notice, Landlord may terminate this Lease
and pursue all remedies available at law.

9. GOVERNING LAW

This Lease is governed by the laws of the State of New York.

Fifth Avenue Holdings LLC              Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Vincent Aldo                     Name: Sarah Chen
Title: Managing Member                 Title: Chief Procurement Officer
"""

LEASE_001_AMENDMENT = """\
AMENDMENT NO. 1 TO COMMERCIAL OFFICE LEASE AGREEMENT

Reference: LEASE-2024-0055-A1
Effective Date of Amendment: March 1, 2026

PARTIES

This Amendment No. 1 (this "Amendment") is entered into as of March 1, 2026, between Fifth
Avenue Holdings LLC, a New York limited liability company ("Landlord"), and Meridian
Analytics Inc., a Delaware corporation ("Tenant").

RECITALS

WHEREAS, the parties entered into that certain Commercial Office Lease Agreement, reference
LEASE-2024-0055, effective June 1, 2024 (the "Lease"); and

WHEREAS, the parties wish to reduce the leased premises and rent, extend the term, and
shorten the renewal-notice period;

NOW, THEREFORE, the parties agree as follows:

1. REDUCTION OF PREMISES

Effective April 1, 2026, the Premises are reduced from approximately 22,000 rentable square
feet to approximately 15,000 rentable square feet, comprising the eastern portion of the
30th floor as shown on Exhibit A-1.  Tenant shall surrender the balance of the 30th floor in
broom-clean condition on or before March 31, 2026.

2. REDUCTION OF RENT

Section 3 (Rent) is amended so that base rent is reduced from USD 780,000 per annum to USD
560,000 per annum, payable in equal monthly installments of USD 46,667, effective April 1,
2026.  The 3% annual escalation continues to apply to the reduced base rent.

3. TERM EXTENSION

Section 2 (Term) is deleted and replaced with the following: "The term commences June 1,
2024 and expires May 31, 2031."

4. RENEWAL NOTICE

Section 4.1 is amended so that the non-renewal notice period is reduced from twelve (12)
months to nine (9) months before the expiry date.

5. NO OTHER CHANGES

Except as expressly amended herein, all terms of the Lease remain in full force and effect.
In the event of any conflict, this Amendment controls.

Fifth Avenue Holdings LLC              Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Vincent Aldo                     Name: Sarah Chen
Title: Managing Member                 Title: Chief Procurement Officer
"""

LEASE_002 = """\
EQUIPMENT LEASE AGREEMENT

Reference: EQL-2023-0288
Effective Date: March 1, 2023
Expiry Date: February 28, 2026

PARTIES

This Equipment Lease Agreement ("Agreement") is entered into between DataCore Leasing Inc.,
a Texas corporation ("Lessor"), and Meridian Analytics Inc., a Delaware corporation
("Lessee").

1. EQUIPMENT

Lessor leases to Lessee the server and networking equipment described in Schedule 1 (the
"Equipment") for installation at Lessee's data-center facility.

2. TERM

The lease term is thirty-six (36) months, commencing March 1, 2023 and ending February 28,
2026.

3. RENT

Lessee shall pay Lessor USD 95,000 in total, invoiced as monthly rent of USD 2,639, payable
in advance on the first business day of each month, net-15.

4. EARLY TERMINATION

Lessee may terminate this Agreement before the end of the term upon sixty (60) days' written
notice, provided that Lessee pays an early-termination charge equal to fifty percent (50%)
of the remaining unpaid rent for the balance of the term.

5. MAINTENANCE AND RISK OF LOSS

Lessee shall maintain the Equipment in good working condition and bears the risk of loss or
damage during the term, ordinary wear and tear excepted.

6. RETURN

Upon expiry or termination, Lessee shall return the Equipment to Lessor in good condition at
Lessee's expense within fifteen (15) days.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of Texas.

DataCore Leasing Inc.                  Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Carla Jimenez                    Name: Sarah Chen
Title: Director of Leasing             Title: Chief Procurement Officer
"""

MSC_001 = """\
MASTER CONSULTING SERVICES AGREEMENT

Reference: MCSA-2024-0611
Effective Date: February 1, 2024
Expiry Date: January 31, 2027

PARTIES

This Master Consulting Services Agreement ("Agreement") is entered into between Meridian
Analytics Inc., a Delaware corporation ("Client"), and Brightpath Consulting LLC, a New York
limited liability company with its principal place of business at 1 Bryant Park, New York, NY
10036 ("Consultant").

1. SERVICES

Consultant shall perform consulting services for Client as described in one or more mutually
executed Statements of Work ("SOWs").  Each SOW is governed by this Agreement and, in the
event of conflict, the terms of this Agreement control except where an SOW expressly states
otherwise.

2. TERM

This Agreement commences February 1, 2024 and continues until January 31, 2027 unless
earlier terminated.  Termination does not affect any SOW then in progress unless the SOW is
also terminated.

3. FEES AND PAYMENT

Fees are set out in each SOW.  Consultant shall invoice Client monthly, and Client shall pay
undisputed amounts within forty-five (45) days of the invoice date (net-45).

4. INTELLECTUAL PROPERTY

Deliverables created specifically for Client under an SOW are assigned to Client upon full
payment.  Consultant retains ownership of its pre-existing materials and general know-how.

5. CONFIDENTIALITY

Each party shall protect the other's Confidential Information for the term and five (5) years
thereafter.

6. LIMITATION OF LIABILITY

6.1  Except as set forth in Section 6.2, each party's aggregate liability under this
Agreement shall not exceed the fees paid or payable under the applicable SOW in the twelve
(12) months preceding the claim.

6.2  The limitation in Section 6.1 shall not apply to a party's breach of its confidentiality
obligations, for which liability is uncapped.

7. TERMINATION

Either party may terminate this Agreement or any SOW for convenience upon thirty (30) days'
written notice, or for cause upon a material breach uncured within thirty (30) days.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of New York.

Meridian Analytics Inc.                Brightpath Consulting LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Laura Devine
Title: Chief Procurement Officer       Title: Managing Partner
"""

SOW_001 = """\
STATEMENT OF WORK NO. 1

Reference: SOW-2024-0611-01
Effective Date: February 15, 2024

This Statement of Work No. 1 ("SOW") is issued under and governed by the Master Consulting
Services Agreement, reference MCSA-2024-0611, dated February 1, 2024 (the "MSA"), between
Meridian Analytics Inc. ("Client") and Brightpath Consulting LLC ("Consultant").
Capitalized terms not defined here have the meanings given in the MSA.

1. SCOPE OF SERVICES

Consultant shall design and deliver a data-warehouse migration plan and reference
implementation, comprising: (a) current-state assessment; (b) target architecture;
(c) migration runbook; and (d) a pilot migration of two production datasets.

2. TERM

The Services under this SOW shall be performed between February 15, 2024 and August 31,
2024.

3. FEES

The total fixed fee for this SOW is USD 220,000, payable on a milestone basis:
  Milestone 1 — Assessment complete: USD 55,000
  Milestone 2 — Target architecture approved: USD 55,000
  Milestone 3 — Migration runbook delivered: USD 55,000
  Milestone 4 — Pilot migration accepted: USD 55,000

Each milestone is invoiced upon Client's written acceptance and paid net-45 in accordance
with the MSA.

4. ACCEPTANCE

Client shall have ten (10) business days to review each deliverable and to accept it or
provide written notice of deficiencies.

5. GOVERNING TERMS

All other terms are governed by the MSA.

Meridian Analytics Inc.                Brightpath Consulting LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Laura Devine
Title: Chief Procurement Officer       Title: Managing Partner
"""

SVC_001 = """\
PROFESSIONAL SERVICES AGREEMENT

Reference: PSA-2024-0740
Effective Date: November 1, 2024
Expiry Date: October 31, 2026

PARTIES

This Professional Services Agreement ("Agreement") is entered into between Meridian Analytics
Inc., a Delaware corporation ("Customer"), and TechVault Solutions, LLC, a California limited
liability company with its principal place of business at 2250 Market Street, San Francisco,
CA 94114 ("Provider").

1. SERVICES

Provider shall provide implementation, integration, and advisory services relating to the
DataSync Enterprise platform, as described in work orders agreed by the parties.

2. TERM

This Agreement commences November 1, 2024 and expires October 31, 2026.

3. FEES AND PAYMENT

Services are billed on a time-and-materials basis at the rates set out in each work order.
Provider shall invoice Customer monthly.  Customer shall pay undisputed amounts within sixty
(60) days of the invoice date (net-60).

4. CONFIDENTIALITY

Each party shall protect the other's Confidential Information for the term and three (3)
years thereafter.

5. LIMITATION OF LIABILITY

PROVIDER'S AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE FEES PAID BY
CUSTOMER IN THE SIX (6) MONTHS PRECEDING THE CLAIM.  NEITHER PARTY SHALL BE LIABLE FOR
INDIRECT OR CONSEQUENTIAL DAMAGES.

6. TERMINATION

Either party may terminate for convenience upon thirty (30) days' written notice, or for
cause upon a material breach uncured within thirty (30) days.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of California.

Meridian Analytics Inc.                TechVault Solutions, LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Priya Mehta
Title: Chief Procurement Officer       Title: General Counsel
"""

SVC_002 = """\
MANAGED IT SERVICES AGREEMENT

Reference: MITS-2023-0521
Effective Date: October 1, 2023
Expiry Date: September 30, 2026

PARTIES

This Managed IT Services Agreement ("Agreement") is entered into between Meridian Analytics
Inc., a Delaware corporation ("Customer"), and Sentinel IT Services Inc., a Colorado
corporation with its principal place of business at 1600 Broadway, Denver, CO 80202
("Provider").

1. SERVICES

Provider shall provide managed IT services, including help-desk support, endpoint
management, patching, and 24x7 monitoring of Customer's corporate IT environment.

2. TERM AND RENEWAL

The initial term runs from October 1, 2023 to September 30, 2026.  Thereafter this Agreement
automatically renews for successive one (1)-year terms unless either party gives written
notice of non-renewal at least ninety (90) days before the end of the then-current term.

3. FEES

Customer shall pay a fixed monthly managed-services fee of USD 18,000, invoiced monthly and
payable net-30.

4. SERVICE LEVELS AND CREDITS

Provider shall respond to critical incidents within one (1) hour and resolve them within
eight (8) hours.  If Provider fails to meet the monthly service-level targets, Customer is
entitled to service credits equal to five percent (5%) of the monthly fee per missed target,
capped at twenty-five percent (25%) of the monthly fee.

5. LIMITATION OF LIABILITY

PROVIDER'S AGGREGATE LIABILITY SHALL NOT EXCEED THE TOTAL FEES PAID IN THE TWELVE (12)
MONTHS PRECEDING THE CLAIM.  SERVICE CREDITS ARE CUSTOMER'S SOLE REMEDY FOR SERVICE-LEVEL
FAILURES.

6. TERMINATION

Either party may terminate for cause upon a material breach uncured within thirty (30) days
of written notice.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of Colorado.

Meridian Analytics Inc.                Sentinel IT Services Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Derek Olsson
Title: Chief Procurement Officer       Title: VP Client Services
"""

MSA_001 = """\
MASTER SERVICES AGREEMENT

Reference: MSA-2024-0466
Effective Date: July 1, 2024
Expiry Date: June 30, 2028

PARTIES

This Master Services Agreement ("Agreement") is entered into between Meridian Analytics Inc.,
a Delaware corporation ("Customer"), and Orion Logistics Inc., an Illinois corporation with
its principal place of business at 233 South Wacker Drive, Chicago, IL 60606 ("Provider").

1. SERVICES

Provider shall provide logistics, fulfillment, and physical-media distribution services for
Customer's hardware appliances, as described in one or more work orders.

2. TERM

This Agreement commences July 1, 2024 and expires June 30, 2028.

3. FEES AND PAYMENT

Fees are set out in each work order.  Provider shall invoice Customer monthly, payable net-30.

4. INDEMNIFICATION

Provider shall indemnify Customer against third-party claims arising from Provider's
negligence, willful misconduct, or breach of this Agreement, including claims for personal
injury or property damage occurring in Provider's facilities.

5. LIMITATION OF LIABILITY

THE PARTIES AGREE THAT NO CONTRACTUAL CAP SHALL APPLY TO EITHER PARTY'S LIABILITY UNDER THIS
AGREEMENT.  EACH PARTY'S LIABILITY IS UNLIMITED AND SHALL BE DETERMINED IN ACCORDANCE WITH
APPLICABLE LAW.  THE PARTIES ACKNOWLEDGE THAT THEY HAVE PRICED THE SERVICES TO REFLECT THIS
ALLOCATION OF RISK.

6. INSURANCE

Provider shall maintain commercial general liability insurance of not less than USD
5,000,000 per occurrence and shall name Customer as an additional insured.

7. TERMINATION

Either party may terminate for convenience upon sixty (60) days' written notice, or for cause
upon a material breach uncured within thirty (30) days.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of Illinois.

Meridian Analytics Inc.                Orion Logistics Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Thomas Reilly
Title: Chief Procurement Officer       Title: Chief Operating Officer
"""

MSA_001_AMENDMENT = """\
AMENDMENT NO. 1 TO MASTER SERVICES AGREEMENT

Reference: MSA-2024-0466-A1
Effective Date of Amendment: January 1, 2026

PARTIES

This Amendment No. 1 (this "Amendment") is entered into as of January 1, 2026, between
Meridian Analytics Inc., a Delaware corporation ("Customer"), and Orion Logistics Inc., an
Illinois corporation ("Provider").

RECITALS

WHEREAS, the parties entered into that certain Master Services Agreement, reference
MSA-2024-0466, effective July 1, 2024 (the "Agreement"), which provided for unlimited
liability with no contractual cap; and

WHEREAS, the parties wish to introduce a limitation of liability;

NOW, THEREFORE, the parties agree as follows:

1. LIMITATION OF LIABILITY

Section 5 (Limitation of Liability) of the Agreement is deleted in its entirety and replaced
with the following:

"5. LIMITATION OF LIABILITY.  EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO
THIS AGREEMENT SHALL NOT EXCEED USD 1,000,000.  THIS CAP SHALL NOT APPLY TO PROVIDER'S
INDEMNIFICATION OBLIGATIONS UNDER SECTION 4, TO A PARTY'S BREACH OF CONFIDENTIALITY, OR TO
LIABILITY ARISING FROM GROSS NEGLIGENCE OR WILLFUL MISCONDUCT.  NEITHER PARTY SHALL BE
LIABLE FOR INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES."

2. NO OTHER CHANGES

Except as expressly amended herein, all terms of the Agreement remain in full force and
effect.  In the event of any conflict, this Amendment controls.

Meridian Analytics Inc.                Orion Logistics Inc.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Thomas Reilly
Title: Chief Procurement Officer       Title: Chief Operating Officer
"""

VND_001 = """\
VENDOR SUPPLY AGREEMENT

Reference: VSA-2023-0198
Effective Date: November 1, 2023
Expiry Date: October 31, 2025

PARTIES

This Vendor Supply Agreement ("Agreement") is entered into between Meridian Analytics Inc.,
a Delaware corporation ("Buyer"), and Pinnacle Hardware Corp., a Washington corporation with
its principal place of business at 500 108th Avenue NE, Bellevue, WA 98004 ("Supplier").

1. SUPPLY OF GOODS

Supplier shall manufacture and supply the server and storage hardware described in Buyer's
purchase orders issued under this Agreement.

2. TERM AND RENEWAL

The initial term runs from November 1, 2023 to October 31, 2025 and automatically renews for
successive one (1)-year terms unless either party provides sixty (60) days' written notice of
non-renewal.

3. PRICING AND PAYMENT

Prices are as set out in the Supplier's price list in effect on the purchase-order date.
Buyer shall pay each invoice net-30.

4. DELIVERY AND TITLE

Supplier shall deliver goods FOB destination.  Title and risk of loss pass to Buyer upon
delivery to Buyer's designated facility.

5. WARRANTY

Supplier warrants that the goods will be free from defects in materials and workmanship for
twelve (12) months from delivery and will conform to the applicable specifications.

6. LIMITATION OF LIABILITY

SUPPLIER'S AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE PURCHASE PRICE OF
THE GOODS GIVING RISE TO THE CLAIM.  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT OR
CONSEQUENTIAL DAMAGES.

7. TERMINATION

Either party may terminate for cause upon a material breach uncured within thirty (30) days
of written notice.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of Washington.

Meridian Analytics Inc.                Pinnacle Hardware Corp.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Angela Ruiz
Title: Chief Procurement Officer       Title: VP Sales
"""

VND_002 = """\
MARKETING SERVICES AGREEMENT

Reference: MKT-2024-0355
Effective Date: May 1, 2024
Expiry Date: April 30, 2026

PARTIES

This Marketing Services Agreement ("Agreement") is entered into between Meridian Analytics
Inc., a Delaware corporation ("Client"), and Vivid Media Group LLC, a Florida limited
liability company with its principal place of business at 1101 Brickell Avenue, Miami, FL
33131 ("Agency").

1. SERVICES

Agency shall provide digital marketing, campaign management, content production, and media
buying services for Client's demand-generation programs.

2. TERM

This Agreement commences May 1, 2024 and expires April 30, 2026.

3. FEES AND PAYMENT

Client shall pay Agency a monthly retainer of USD 25,000 plus approved media spend billed at
cost.  Invoices are issued monthly and payable within thirty (30) days.

4. INTELLECTUAL PROPERTY

Upon full payment, Agency assigns to Client all deliverables created specifically for Client.
Agency retains ownership of its tools, templates, and general methodologies.

5. CONFIDENTIALITY

Each party shall protect the other's Confidential Information for the term and two (2) years
thereafter.

6. TERMINATION

Either party may terminate for convenience upon sixty (60) days' written notice, or for cause
upon a material breach uncured within thirty (30) days.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of Florida.

Meridian Analytics Inc.                Vivid Media Group LLC
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Isabella Cruz
Title: Chief Marketing Officer         Title: Managing Director
"""

DPA_001 = """\
DATA PROCESSING ADDENDUM

Reference: DPA-2024-0391-01
Effective Date: April 1, 2024

PARTIES

This Data Processing Addendum ("DPA") supplements and forms part of the Software as a Service
Agreement No. MSA-2024-0391, effective April 1, 2024 (the "Principal Agreement"), between
Meridian Analytics Inc., a Delaware corporation ("Controller"), and Acme Corp, a New York
corporation ("Processor").  Capitalized terms not defined herein have the meanings given in
the Principal Agreement.

1. SCOPE AND ROLES

The parties acknowledge that, in providing the CloudStore Pro services, Processor processes
personal data on behalf of Controller.  Processor acts as a data processor and Controller as
a data controller within the meaning of Regulation (EU) 2016/679 ("GDPR").

2. PROCESSING INSTRUCTIONS

Processor shall process personal data only on documented instructions from Controller,
including with regard to transfers of personal data to a third country, unless required to do
so by applicable law.

3. SECURITY MEASURES

Processor shall implement appropriate technical and organisational measures to ensure a
level of security appropriate to the risk, including encryption of personal data in transit
and at rest, ongoing confidentiality and integrity of processing systems, and regular
testing of security measures.

4. SUBPROCESSORS

Processor shall not engage a subprocessor without Controller's prior written authorisation.
Processor shall maintain a current list of authorised subprocessors, attached as Annex 2, and
shall give Controller at least thirty (30) days' notice of any intended addition or
replacement, during which Controller may object on reasonable data-protection grounds.

5. INTERNATIONAL TRANSFERS

Any transfer of personal data outside the European Economic Area shall be subject to
appropriate safeguards under Chapter V of the GDPR, including the EU Standard Contractual
Clauses, which are incorporated herein by reference and completed in Annex 3.

6. DATA SUBJECT RIGHTS AND BREACH

Processor shall assist Controller in responding to data-subject requests and shall notify
Controller without undue delay, and in any event within forty-eight (48) hours, after
becoming aware of a personal data breach.

7. DELETION AND AUDIT

Upon termination of the Principal Agreement, Processor shall delete or return all personal
data.  Processor shall make available information necessary to demonstrate compliance and
allow for audits by Controller or its mandated auditor.

8. GOVERNING LAW

This DPA is governed by the same law as the Principal Agreement, namely the laws of the State
of New York, except that the Standard Contractual Clauses are governed by their own terms.

Meridian Analytics Inc.                Acme Corp
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Michael Torres
Title: Data Protection Officer         Title: VP Enterprise Sales
"""

LIC_001 = """\
PERPETUAL SOFTWARE LICENSE AGREEMENT

Reference: LIC-2022-0140
Effective Date: August 1, 2022

PARTIES

This Perpetual Software License Agreement ("Agreement") is entered into between Quantum
Software Labs Inc., a Massachusetts corporation with its principal place of business at 200
Clarendon Street, Boston, MA 02116 ("Licensor"), and Meridian Analytics Inc., a Delaware
corporation ("Licensee").

1. LICENSE GRANT

Licensor grants Licensee a perpetual, non-exclusive, non-transferable license to install and
use the object code of the QuantumCompute on-premises software (the "Software") for
Licensee's internal business operations, on up to sixteen (16) server nodes.

2. TERM

The license granted in Section 1 is perpetual and has no expiration date, subject to Section
7.  The maintenance and support services in Section 4 renew annually as provided therein.

3. LICENSE FEE

Licensee shall pay a one-time perpetual license fee of USD 600,000, due within thirty (30)
days of the Effective Date.

4. MAINTENANCE AND SUPPORT

Licensee shall pay an annual maintenance and support fee of USD 150,000, invoiced each
anniversary of the Effective Date, entitling Licensee to updates, upgrades, and technical
support.  Licensee may decline to renew maintenance without affecting the perpetual license.

5. LIMITATION OF LIABILITY

LICENSOR'S AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE ONE-TIME LICENSE
FEE PAID UNDER SECTION 3.  NEITHER PARTY SHALL BE LIABLE FOR INDIRECT OR CONSEQUENTIAL
DAMAGES.

6. INTELLECTUAL PROPERTY

The Software is licensed, not sold.  Licensor retains all right, title, and interest in the
Software and all intellectual property therein.

7. TERMINATION

Licensor may terminate the license only if Licensee materially breaches the license scope or
confidentiality obligations and fails to cure within thirty (30) days of written notice.

8. GOVERNING LAW

This Agreement is governed by the laws of the Commonwealth of Massachusetts.

Quantum Software Labs Inc.             Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Nina Kapoor                      Name: Sarah Chen
Title: General Counsel                 Title: Chief Procurement Officer
"""

RSL_001 = """\
RESELLER AND CHANNEL PARTNER AGREEMENT

Reference: RSL-2024-0777
Effective Date: October 1, 2024
Expiry Date: September 30, 2027

PARTIES

This Reseller and Channel Partner Agreement ("Agreement") is entered into between Northstar
Analytics Inc., a Georgia corporation with its principal place of business at 1180 Peachtree
Street NE, Atlanta, GA 30309 ("Vendor"), and Meridian Analytics Inc., a Delaware corporation
("Reseller").

1. APPOINTMENT

Vendor appoints Reseller as an authorized reseller of the Northstar analytics product suite.
Reseller's appointment is exclusive within the territory of the Northeastern United States
(the "Territory") for the term of this Agreement.

2. TERM AND RENEWAL

The initial term runs from October 1, 2024 to September 30, 2027 and automatically renews for
successive one (1)-year terms unless either party provides written notice of non-renewal at
least ninety (90) days before the end of the then-current term.

3. PRICING AND MARGIN

Reseller may purchase Vendor products at a discount of twenty percent (20%) off Vendor's
then-current list price and may resell them at prices Reseller determines.  Reseller shall
pay Vendor net-45 on all orders.

4. OBLIGATIONS OF RESELLER

Reseller shall use commercially reasonable efforts to market and sell the products, maintain
trained sales staff, and comply with Vendor's brand guidelines.

5. LIMITATION OF LIABILITY

EACH PARTY'S AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE FEES PAID OR
PAYABLE BY RESELLER TO VENDOR IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

6. TERMINATION

Either party may terminate for cause upon a material breach uncured within thirty (30) days.
Upon termination, Reseller's exclusive rights in the Territory cease immediately.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of Georgia.

Northstar Analytics Inc.               Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Marcus Webb                      Name: Sarah Chen
Title: Chief Revenue Officer           Title: Chief Procurement Officer
"""

GRN_001 = """\
SOFTWARE EVALUATION AGREEMENT

Reference: EVAL-2025-0061
Effective Date: June 1, 2025
Expiry Date: August 31, 2025

PARTIES

This Software Evaluation Agreement ("Agreement") is entered into between Beacon AI Inc., a
California corporation with its principal place of business at 405 Howard Street, San
Francisco, CA 94105 ("Provider"), and Meridian Analytics Inc., a Delaware corporation
("Evaluator").

1. EVALUATION LICENSE

Provider grants Evaluator a limited, non-exclusive, non-transferable license to access the
Beacon AI platform solely for internal evaluation during the Evaluation Period.

2. EVALUATION PERIOD AND CONVERSION

2.1  The Evaluation Period is ninety (90) days, commencing June 1, 2025 and ending August 31,
2025.

2.2  Unless Evaluator gives written notice of non-conversion at least ten (10) days before the
end of the Evaluation Period, this Agreement shall automatically convert into a paid
twelve (12)-month subscription at Provider's then-current list price of USD 96,000 per year,
billed annually and payable net-30.

3. NO FEES DURING EVALUATION

No fees are payable during the Evaluation Period unless and until conversion occurs under
Section 2.2.

4. LIMITATION OF LIABILITY

DURING THE EVALUATION PERIOD, THE PLATFORM IS PROVIDED "AS IS" WITHOUT WARRANTY, AND
PROVIDER'S AGGREGATE LIABILITY SHALL NOT EXCEED USD 5,000.

5. CONFIDENTIALITY

Each party shall protect the other's Confidential Information, including Provider's platform
and any benchmark or evaluation results, for three (3) years.

6. GOVERNING LAW

This Agreement is governed by the laws of the State of California.

Beacon AI Inc.                         Meridian Analytics Inc.
By: ____________________________       By: ____________________________
Name: Sofia Alvarez                    Name: Sarah Chen
Title: Head of Partnerships            Title: Chief Procurement Officer
"""

PUR_001 = """\
MASTER PURCHASE AGREEMENT

Reference: MPA-2024-0512
Effective Date: June 1, 2024
Expiry Date: May 31, 2027

PARTIES

This Master Purchase Agreement ("Agreement") is entered into between Meridian Analytics Inc.,
a Delaware corporation ("Buyer"), and Ironclad Manufacturing Co., an Ohio corporation with
its principal place of business at 600 Superior Avenue, Cleveland, OH 44114 ("Seller").

1. PURCHASE OF PRODUCTS

Seller shall manufacture and sell to Buyer the ruggedized enclosure and rack products
described in purchase orders issued under this Agreement.  Each accepted purchase order forms
a separate contract incorporating these terms.

2. TERM

This Agreement commences June 1, 2024 and expires May 31, 2027.

3. PRICE AND PAYMENT

Prices are as quoted by Seller and accepted in each purchase order.  Buyer shall pay each
invoice within thirty (30) days of delivery (net-30).

4. DELIVERY

Seller shall deliver products in accordance with the delivery schedule in each purchase
order, FOB Seller's plant.  Time is of the essence for delivery.

5. WARRANTY

Seller warrants that products will conform to specifications and be free from defects in
materials and workmanship for eighteen (18) months from delivery.

6. LIMITATION OF LIABILITY

SELLER'S AGGREGATE LIABILITY SHALL NOT EXCEED THE AGGREGATE PURCHASE PRICE PAID BY BUYER
UNDER THE PURCHASE ORDER GIVING RISE TO THE CLAIM.  NEITHER PARTY SHALL BE LIABLE FOR
INDIRECT OR CONSEQUENTIAL DAMAGES.

7. TERMINATION

Either party may terminate for cause upon a material breach uncured within thirty (30) days
of written notice.

8. GOVERNING LAW

This Agreement is governed by the laws of the State of Ohio.

Meridian Analytics Inc.                Ironclad Manufacturing Co.
By: ____________________________       By: ____________________________
Name: Sarah Chen                       Name: Walter Boyd
Title: Chief Procurement Officer       Title: VP Sales
"""


# Mapping from doc_id to contract text — use this in tests.  The first six
# fixtures (saas-*) are asserted field-by-field in test_queries.py and must not
# be edited; the rest form the extended portfolio (see the module docstring for
# the planted cross-document signals the hero queries exploit).
CONTRACTS: dict[str, str] = {
    # --- core SaaS set (asserted in test_queries.py) ---
    "saas-001": SAAS_001,
    "saas-001-amendment": SAAS_001_AMENDMENT,
    "saas-002": SAAS_002,
    "saas-003": SAAS_003,
    "saas-004": SAAS_004,
    "saas-005": SAAS_005,
    # --- extended portfolio ---
    "saas-006": SAAS_006,
    "nda-001": NDA_001,
    "nda-002": NDA_002,
    "nda-003": NDA_003,
    "emp-001": EMP_001,
    "emp-002": EMP_002,
    "emp-003": EMP_003,
    "sep-001": SEP_001,
    "lease-001": LEASE_001,
    "lease-001-amendment": LEASE_001_AMENDMENT,
    "lease-002": LEASE_002,
    "msc-001": MSC_001,
    "sow-001": SOW_001,
    "svc-001": SVC_001,
    "svc-002": SVC_002,
    "msa-001": MSA_001,
    "msa-001-amendment": MSA_001_AMENDMENT,
    "vnd-001": VND_001,
    "vnd-002": VND_002,
    "dpa-001": DPA_001,
    "lic-001": LIC_001,
    "rsl-001": RSL_001,
    "grn-001": GRN_001,
    "pur-001": PUR_001,
}
