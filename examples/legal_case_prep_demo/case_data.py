"""Sample case bundle for the legal case preparation demo.

Fictional dispute — Acme Industrial Supplies Ltd v Beacon Manufacturing PLC —
deliberately constructed so the pipeline produces real cross-document
contradictions and evidence gaps.

The bundle covers nine documents that a lawyer would typically receive at
the start of a commercial dispute: a supply contract, two emails, a delivery
note, a witness statement, a solicitor's letter, a notice of termination, an
expert report, and the particulars of claim.

Expected findings summary
─────────────────────────
delivery-dispute
  - Acme's signed delivery note claims 200 valves delivered and accepted on
    14 March 2025 (doc-003). Beacon's warehouse manager states in a witness
    statement that 180 valves arrived on 16 March 2025, with 20 missing
    (doc-004). → contradiction (critical)
  - Acme's particulars of claim repeat the 14 March / 200 valves figure;
    Beacon's solicitor letter records "approximately 180 units" received
    around 16 March.

defect-allegation
  - Beacon's solicitor letter dated 5 April 2025 claims five valves were
    visibly damaged on arrival (doc-006). The expert report dated
    1 July 2025 finds four valves with manufacturing defects (doc-008).
    → contradiction (medium)
  - No contemporaneous photographs or inspection record corroborating the
    "visibly damaged" claim — evidence gap.

payment-default / breach-notice
  - Acme's termination notice (doc-007) recites that Beacon failed to pay
    by 30 April 2025. Beacon disputes that payment was due because delivery
    was incomplete (doc-006). → contradiction on whether the payment was due.
  - No bank statement or independent confirmation of any payment having
    been tendered — evidence gap on Beacon's side.
"""

from cogbase.core.models import Document

_META = {"case_id": "ACME-BEACON-2025"}


CASE_DOCUMENTS: list[Document] = [

    # ------------------------------------------------------------------
    # doc-001 — Supply Agreement (contract)
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-001",
        metadata={**_META, "source": "supply_agreement_2025_02_01.pdf",
                  "doc_format": "pdf"},
        text="""\
SUPPLY AGREEMENT

This Supply Agreement ("Agreement") is made on 1 February 2025 between:

  (1) ACME INDUSTRIAL SUPPLIES LIMITED, a company incorporated in England
      and Wales (company no. 06321457), whose registered office is at
      14 Foundry Road, Sheffield S9 3LT ("the Supplier"); and

  (2) BEACON MANUFACTURING PLC, a company incorporated in England and Wales
      (company no. 04219800), whose registered office is at 2 Beacon Way,
      Coventry CV5 6FE ("the Buyer").

1. SUBJECT MATTER

1.1  The Supplier shall manufacture and deliver to the Buyer 200 (two hundred)
     ML-7 industrial valves conforming to the technical specification at
     Schedule 1 ("the Goods").

1.2  The unit price is £450 (four hundred and fifty pounds). The total
     contract price is £90,000 (ninety thousand pounds), exclusive of VAT.

2. DELIVERY

2.1  The Supplier shall deliver the Goods to the Buyer's facility at
     2 Beacon Way, Coventry CV5 6FE on or before 15 March 2025
     ("the Delivery Date").

2.2  Delivery is effected when the Goods are received at the Buyer's facility
     and signed for by a person authorised by the Buyer. The Buyer shall
     inspect the Goods within five (5) business days of delivery and notify
     the Supplier in writing of any shortage, damage, or non-conformance.

2.3  Goods not rejected in writing within the five business day period shall
     be deemed accepted.

3. PAYMENT

3.1  The Buyer shall pay the contract price within 30 days of the date on
     which the Goods are accepted under clause 2.3 ("the Payment Due Date").

3.2  Amounts not paid by the Payment Due Date shall accrue interest at 4%
     per annum above the Bank of England base rate.

4. TERMINATION

4.1  Either party may terminate this Agreement by written notice if the other
     party commits a material breach which is not remedied within 14 days of
     written notice specifying the breach.

5. GOVERNING LAW AND JURISDICTION

5.1  This Agreement is governed by the laws of England and Wales.

5.2  The parties submit to the exclusive jurisdiction of the courts of
     England and Wales.

SIGNED for and on behalf of ACME INDUSTRIAL SUPPLIES LIMITED
    by Michael Chen, Managing Director, on 1 February 2025.

SIGNED for and on behalf of BEACON MANUFACTURING PLC
    by Diana Cole, Chief Executive Officer, on 1 February 2025.
""",
    ),

    # ------------------------------------------------------------------
    # doc-002 — Email from Beacon to Acme confirming order
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-002",
        metadata={**_META, "source": "email_patel_to_acme_2025_02_08.eml",
                  "doc_format": "email"},
        text="""\
From: Sarah Patel <s.patel@beacon-mfg.co.uk>
To: orders@acme-industrial.co.uk
Cc: Diana Cole <d.cole@beacon-mfg.co.uk>
Date: Saturday, 8 February 2025 09:14
Subject: PO-2025-0218 — Confirmation of order, ML-7 valves

Dear Acme team,

Please find attached our purchase order PO-2025-0218 against the Supply
Agreement signed on 1 February 2025. The order is for 200 ML-7 industrial
valves at £450 each, total £90,000 exclusive of VAT.

Per clause 2.1 of the Agreement, delivery is required on or before
15 March 2025 to our Coventry facility. Please confirm the dispatch date
by return.

Diana Cole (our CEO) is copied for visibility. John Reid in our warehouse
will sign for the goods on the day; he is the only person authorised to
acknowledge receipt on Beacon's behalf.

Kind regards,
Sarah Patel
Head of Procurement
Beacon Manufacturing PLC
""",
    ),

    # ------------------------------------------------------------------
    # doc-003 — Acme delivery note signed by Beacon warehouse
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-003",
        metadata={**_META, "source": "delivery_note_AC-DN-3318_2025_03_14.pdf",
                  "doc_format": "pdf"},
        text="""\
ACME INDUSTRIAL SUPPLIES LIMITED
14 Foundry Road, Sheffield S9 3LT

DELIVERY NOTE
Reference: AC-DN-3318
Date of delivery: 14 March 2025
Customer: Beacon Manufacturing PLC, 2 Beacon Way, Coventry CV5 6FE
Purchase order: PO-2025-0218
Supply Agreement: dated 1 February 2025

Goods supplied:
  Item: ML-7 industrial valve (per Schedule 1 specification)
  Quantity: 200 units
  Unit price: £450
  Total: £90,000 (excl. VAT)

The carrier (Midland Heavy Haulage Ltd, vehicle registration MH62 RXP)
delivered the consignment to the Buyer's facility at 11:42 on 14 March 2025.
All 200 units were unloaded, counted, and received without observation of
any shortage or visible damage.

Received and accepted by:
  Signature: J. Reid
  Name (block capitals): JOHN REID
  Position: Warehouse Supervisor
  Date: 14 March 2025
""",
    ),

    # ------------------------------------------------------------------
    # doc-004 — Witness statement, Sarah Patel
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-004",
        metadata={**_META, "source": "witness_statement_patel_2025_05_12.docx",
                  "doc_format": "word"},
        text="""\
IN THE HIGH COURT OF JUSTICE
BUSINESS AND PROPERTY COURTS

Claim No. BL-2025-000642

Between
  ACME INDUSTRIAL SUPPLIES LIMITED                                 Claimant
                                  - and -
  BEACON MANUFACTURING PLC                                       Defendant

WITNESS STATEMENT OF SARAH PATEL

I, SARAH PATEL, of 2 Beacon Way, Coventry CV5 6FE, Head of Procurement at
Beacon Manufacturing PLC, will say as follows:

1.  I am the Head of Procurement at the Defendant company. I make this
    statement from my own knowledge save where indicated otherwise. The
    facts in this statement are true to the best of my knowledge and belief.

2.  I have been responsible for the procurement of the ML-7 industrial
    valves the subject of these proceedings since the negotiations began in
    late January 2025.

3.  The Supply Agreement was signed on 1 February 2025 and provided for
    delivery of 200 valves on or before 15 March 2025.

4.  Acme's lorry did not arrive at our Coventry facility on 14 March 2025
    as Acme now alleges. It arrived shortly after 16:00 on 16 March 2025.
    I was on site at the time of delivery and I personally counted the
    units as they were unloaded.

5.  Only 180 valves were delivered. Twenty valves were missing. A further
    five of the 180 delivered units had visible damage to the actuator
    housing, which I photographed at the time.

6.  John Reid, who signed the carrier's delivery note, did so under
    pressure from the lorry driver, who refused to wait while the goods
    were counted. Mr Reid signed only to acknowledge receipt of the
    consignment, not its conformity. He did not count the units.

7.  I notified Acme of the shortage and damage by telephone on the
    afternoon of 17 March 2025, speaking to Michael Chen.

Statement of truth

I believe that the facts stated in this witness statement are true.

Signed: Sarah Patel
Date: 12 May 2025
""",
    ),

    # ------------------------------------------------------------------
    # doc-005 — Email from Acme MD to Beacon CEO demanding payment
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-005",
        metadata={**_META, "source": "email_chen_to_cole_2025_03_25.eml",
                  "doc_format": "email"},
        text="""\
From: Michael Chen <m.chen@acme-industrial.co.uk>
To: Diana Cole <d.cole@beacon-mfg.co.uk>
Cc: accounts@acme-industrial.co.uk
Date: Tuesday, 25 March 2025 17:48
Subject: Invoice AC-INV-7741 — ML-7 valves, payment due 30 April 2025

Diana,

Following delivery of the 200 ML-7 valves to your Coventry facility on
14 March 2025, signed for by your Warehouse Supervisor John Reid, the
five business day inspection window under clause 2.2 of our Supply
Agreement closed on 21 March 2025 without any written notice of shortage
or non-conformance. The Goods are therefore deemed accepted under
clause 2.3.

The 30-day payment period under clause 3.1 began on 21 March 2025. The
sum of £90,000 (excl. VAT) is payable on or before 30 April 2025.
Invoice AC-INV-7741 is attached.

I expect to see funds cleared by close of business on 30 April. If we
do not, we will treat the non-payment as a material breach and follow
the process at clause 4.1.

Regards,
Michael Chen
Managing Director
Acme Industrial Supplies Limited
""",
    ),

    # ------------------------------------------------------------------
    # doc-006 — Solicitor letter from Hartfield & Co (Beacon)
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-006",
        metadata={**_META, "source": "hartfield_letter_2025_04_05.pdf",
                  "doc_format": "pdf"},
        text="""\
HARTFIELD & CO SOLICITORS
17 Bedford Row, London WC1R 4HE

Our ref: DJH/BMP/2025-114
Date: 5 April 2025

By email and recorded delivery
Michael Chen
Managing Director
Acme Industrial Supplies Limited
14 Foundry Road
Sheffield S9 3LT

Dear Sir,

Re: ML-7 industrial valves — Supply Agreement dated 1 February 2025

We act for Beacon Manufacturing PLC ("our client") in connection with the
captioned agreement. This letter responds to your email to Ms Diana Cole
of 25 March 2025.

1.  Delivery did not occur on 14 March 2025. The Acme consignment arrived
    at our client's Coventry facility late in the afternoon of 16 March
    2025, one day after the contractual Delivery Date under clause 2.1.

2.  Approximately 180 units were delivered, not the 200 units required by
    the Agreement. Our client's records show a shortfall of 20 valves.

3.  Of the 180 units delivered, five exhibited visible damage to the
    actuator housing at the time of unloading. Photographs were taken by
    Ms Sarah Patel, Head of Procurement.

4.  No inspection certificate or test report was provided with the
    consignment, contrary to the practice contemplated by Schedule 1.

5.  Mr John Reid, the warehouse supervisor, signed the carrier's delivery
    note only as a receipt for the consignment. He did not count the units
    or accept the Goods as conforming. He has no authority to do so. The
    only person authorised to acknowledge acceptance on Beacon's behalf is
    Ms Patel.

6.  In the circumstances, the Goods have not been accepted under clause
    2.3 and no payment obligation under clause 3.1 has yet arisen. Our
    client is willing to engage in good-faith negotiation on a pro-rated
    basis for the 175 conforming units, subject to verification.

We look forward to your substantive response within 14 days.

Yours faithfully,
HARTFIELD & CO
Daniel J. Hartfield, Partner
""",
    ),

    # ------------------------------------------------------------------
    # doc-007 — Acme's Notice of Termination
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-007",
        metadata={**_META, "source": "termination_notice_acme_2025_05_10.pdf",
                  "doc_format": "pdf"},
        text="""\
ACME INDUSTRIAL SUPPLIES LIMITED
14 Foundry Road, Sheffield S9 3LT

NOTICE OF TERMINATION
Date: 10 May 2025

To:
Diana Cole
Chief Executive Officer
Beacon Manufacturing PLC
2 Beacon Way
Coventry CV5 6FE

By email and recorded delivery

Re: Supply Agreement dated 1 February 2025 — termination for non-payment

1.  Acme Industrial Supplies Limited ("Acme") delivered 200 ML-7
    industrial valves to Beacon Manufacturing PLC ("Beacon") on
    14 March 2025. Beacon's Warehouse Supervisor, Mr John Reid, signed
    the delivery note (AC-DN-3318) acknowledging receipt of all 200
    units in good condition.

2.  No written notice of any shortage, damage, or non-conformance was
    served within five business days, and the Goods were deemed accepted
    on 21 March 2025 under clause 2.3 of the Supply Agreement.

3.  The sum of £90,000 (excl. VAT) was payable on or before 30 April
    2025 under clause 3.1. No payment has been received.

4.  By a letter from Hartfield & Co Solicitors dated 5 April 2025
    Beacon asserted (incorrectly) that delivery had not occurred until
    16 March 2025 and that the consignment was short and damaged. Those
    allegations were raised more than five business days after delivery
    and are out of time under clause 2.3.

5.  Acme treats Beacon's non-payment as a material breach of the
    Agreement which has not been remedied within 14 days of written
    notice (Acme's email of 25 March 2025).

6.  In accordance with clause 4.1, Acme HEREBY GIVES NOTICE that the
    Supply Agreement is terminated with immediate effect. Acme reserves
    all of its rights to recover the contract price, interest under
    clause 3.2, and the costs of any further proceedings.

Signed:
Michael Chen
Managing Director
Acme Industrial Supplies Limited
""",
    ),

    # ------------------------------------------------------------------
    # doc-008 — Expert report
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-008",
        metadata={**_META, "source": "expert_report_vance_2025_07_01.pdf",
                  "doc_format": "pdf"},
        text="""\
EXPERT REPORT

In the matter of
ACME INDUSTRIAL SUPPLIES LIMITED v BEACON MANUFACTURING PLC
Claim No. BL-2025-000642

Prepared by: Dr Helena Vance, CEng FIMechE
Instructed by: Hartfield & Co Solicitors (on behalf of the Defendant)
Date of report: 1 July 2025

Instructions

1.  I was instructed on 30 May 2025 to inspect the ML-7 industrial valves
    held at the Defendant's Coventry facility and to opine on whether
    they conform to the technical specification at Schedule 1 of the
    Supply Agreement dated 1 February 2025.

Inventory taken on inspection

2.  On 18 June 2025 I attended at 2 Beacon Way, Coventry and counted
    175 ML-7 valves stored in bays 4 to 9 of the Defendant's warehouse.
    No further valves were located on site.

Findings on conformity

3.  Of the 175 valves inspected I identified four valves with manufacturing
    defects to the actuator housing inconsistent with the Schedule 1
    specification. Serial numbers of the four valves are listed at
    Appendix A.

4.  The remaining 171 valves conform to the Schedule 1 specification.

5.  I was not asked to opine on, and I express no opinion on, the
    circumstances of delivery, the number of valves originally dispatched,
    or the number of valves visibly damaged at the point of unloading.

Statement of independence

6.  I confirm that this report has been prepared in accordance with my
    duty to the Court and is independent of the parties.

Signed: Dr Helena Vance
1 July 2025
""",
    ),

    # ------------------------------------------------------------------
    # doc-009 — Particulars of Claim
    # ------------------------------------------------------------------
    Document(
        doc_id="doc-009",
        metadata={**_META, "source": "particulars_of_claim_2025_07_30.pdf",
                  "doc_format": "pdf"},
        text="""\
IN THE HIGH COURT OF JUSTICE
BUSINESS AND PROPERTY COURTS

Claim No. BL-2025-000642

Between
  ACME INDUSTRIAL SUPPLIES LIMITED                                 Claimant
                                  - and -
  BEACON MANUFACTURING PLC                                       Defendant

PARTICULARS OF CLAIM

1.  The Claimant is and was at all material times a company incorporated
    in England and Wales carrying on business as a manufacturer of
    industrial valves. The Defendant is and was at all material times a
    company incorporated in England and Wales carrying on business as a
    manufacturer of industrial machinery.

2.  By a written Supply Agreement dated 1 February 2025 between the
    Claimant as supplier and the Defendant as buyer, the Claimant agreed
    to supply 200 ML-7 industrial valves at a total contract price of
    £90,000 (excl. VAT), to be delivered on or before 15 March 2025.

3.  On 14 March 2025 the Claimant delivered all 200 valves to the
    Defendant's facility at 2 Beacon Way, Coventry. The Defendant's
    Warehouse Supervisor, Mr John Reid, signed the Claimant's delivery
    note acknowledging receipt of the full consignment in good condition.

4.  The Defendant gave no written notice of any shortage, damage, or
    non-conformance within five business days of delivery as required by
    clause 2.2 of the Supply Agreement. The Goods were accordingly
    accepted on 21 March 2025 under clause 2.3.

5.  The contract price of £90,000 (excl. VAT) became payable on or before
    30 April 2025 under clause 3.1. No part of that sum has been paid.

6.  By a Notice of Termination dated 10 May 2025 the Claimant terminated
    the Agreement under clause 4.1 for the Defendant's material breach
    of the payment obligation.

7.  The Claimant accordingly claims:

    (a) the sum of £90,000 (excl. VAT);
    (b) interest pursuant to clause 3.2 of the Supply Agreement from
        1 May 2025 until payment;
    (c) costs.

Statement of truth

The Claimant believes that the facts stated in these Particulars of Claim
are true.

Signed for and on behalf of the Claimant
Michael Chen, Managing Director
Date: 30 July 2025
""",
    ),
]
