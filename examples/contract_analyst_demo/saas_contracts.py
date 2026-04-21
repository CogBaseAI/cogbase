"""Realistic SaaS vendor agreement fixtures for end-to-end testing.

Five SaaS contracts are included, each deliberately crafted to exercise a
specific query pattern or cross-contract comparison:

  saas-001  CloudStore Pro / Acme Corp — NY law, expires 2025-06-30, very low
            liability cap ($50 K) relative to contract value ($500 K), GDPR +
            breach notification language, net-30 payment.

  saas-002  DataSync Enterprise / TechVault Solutions — CA law, expires
            2026-06-30, liability cap equals annual value ($240 K), audit
            rights language, monthly payment.

  saas-003  SecureVault Platform / Nexus Security Ltd — DE law, expires
            2025-12-31, high liability cap ($2 M) reflecting a security-first
            vendor, breach notification + GDPR + audit rights.

  saas-004  AnalyticsPro Suite / Acme Corp — NY law, expires 2027-03-31,
            low liability cap ($250 K) vs high value ($1.2 M), upfront payment
            (contradicts saas-001's net-30 for the same vendor), auto-renewal
            clause (90-day trigger).

  saas-005  WorkflowManager Pro / Apex Systems — TX law, expires 2025-09-30,
            unusually long 180-day termination notice, assignment-to-competitors
            restriction.

Cross-contract signals deliberately embedded
  Pattern A  3 contracts expire before 2026-01-01 (saas-001, 003, 005)
             2 contracts governed by New York law (saas-001, 004)
             Acme Corp is a party in saas-001 and saas-004
             Only saas-003 has a liability cap above $1 M
  Pattern B  GDPR / data residency in saas-001 and saas-003, saas-004
             Breach notification in saas-001 and saas-003
             Audit rights in saas-002 and saas-003
             Competitor assignment restriction only in saas-005
  Pattern C  saas-001 (net-30) vs saas-004 (upfront) contradict for Acme Corp
             saas-005 notice period (180 days) far exceeds the others (30–90)
             saas-001 cap ($50 K) / value ($500 K) and saas-004 cap ($250 K) /
             value ($1.2 M) are outliers
  Pattern D  All five have termination, governing law, and dispute resolution
             saas-004 auto-renewal trigger 90 days before 2027-03-31
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

# Mapping from doc_id to contract text — use this in tests
CONTRACTS: dict[str, str] = {
    "saas-001": SAAS_001,
    "saas-002": SAAS_002,
    "saas-003": SAAS_003,
    "saas-004": SAAS_004,
    "saas-005": SAAS_005,
}
