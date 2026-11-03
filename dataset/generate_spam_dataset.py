"""
Spam & Ham Email Dataset Generator
Creates a realistic, diverse dataset of emails for training and evaluating
spam classification models (Naive Bayes vs. Support Vector Machines).
"""

import os
import random
import pandas as pd

random.seed(42)

# Templates for legitimate emails (HAM)
HAM_TEMPLATES = [
    ("Team meeting rescheduled to {time}", "Hi team,\nPlease note that our weekly sync has been moved to {time}. Let me know if you have conflicting meetings.\nBest regards,\n{name}"),
    ("Code review requested: PR #{pr_num}", "Hello {name},\nCould you please take a look at PR #{pr_num} when you have some time? It addresses the database connection pool leak and adds unit tests.\nThanks,\nDev Team"),
    ("Sprint planning notes & action items", "Hi everyone,\nHere are the notes from today's sprint planning:\n- Finalize API schema by Thursday\n- Bug bash scheduled for Friday 2 PM\n- Target release date: next Tuesday.\nThanks!"),
    ("Monthly cloud infrastructure report", "Hello,\nYour monthly AWS/GCP usage report for account {acc_num} is ready for review. Total spend was within budget at ${amount}.\nDetailed breakdown attached."),
    ("Dinner tonight?", "Hey {name},\nAre we still on for dinner around 7 PM? Let me know if that spot on 5th street works for you.\nCheers!"),
    ("Invoice #{inv_num} payment confirmation", "Dear Customer,\nWe have received your payment of ${amount} for invoice #{inv_num}. Thank you for your continued business.\nAccounting Dept."),
    ("Antigravity platform update v2.4 released", "Hi Developer,\nWe are excited to share new features in v2.4, including improved subagent communication, faster execution sandboxes, and upgraded metrics export.\nRead the release notes online."),
    ("Security alert: New login from Chrome on Windows", "Hi {name},\nA new login was detected for your account from IP 192.168.1.{ip_suffix} (Windows / Chrome). If this was you, you can safely ignore this message. Otherwise, review your active sessions immediately."),
    ("Q3 financial summary presentation slides", "Hi Leadership,\nAttached are the finalized slides for the Q3 financial review. Overall revenue increased by 14% year-over-year. Please review before tomorrow's board meeting.\nRegards,\nFinance Team"),
    ("Flight confirmation - Booking reference #{booking_ref}", "Dear Passenger,\nYour flight from SFO to JFK on {date} is confirmed. Booking reference: {booking_ref}. Check-in opens 24 hours prior to departure."),
    ("Doctor appointment reminder", "Hello,\nThis is a reminder for your upcoming appointment with Dr. Smith on {date} at 10:30 AM. Reply CONFIRM to confirm or CALL to reschedule."),
    ("Feedback on proposed system architecture", "Hi {name},\nI reviewed the RFC for microservice decoupling. The event-driven architecture looks solid, but we should add a dead-letter queue for failed Kafka events.\nBest,\nLead Architect"),
    ("Library book due date reminder", "Dear Patron,\nYour loan of 'Introduction to Machine Learning' is due on {date}. You can renew online via your student/patron portal."),
    ("Welcome to our engineering newsletter!", "Hi there,\nIn this week's edition, we explore vector databases, optimization techniques in PyTorch, and clean code principles. Enjoy reading!"),
    ("Project timeline update and milestones", "Hi team,\nPhase 1 deliverables have been accepted by client. We are proceeding to Phase 2 next Monday. Great job everyone!"),
    ("Your package has been delivered to your front door", "Hello,\nYour delivery driver has dropped off package #{pkg_num}. Please inspect the contents and let customer support know if anything is missing.\nTrack package details in your account."),
    ("Database backup verification successful", "System Automated Alert: Daily incremental backup for cluster-us-east-1 completed successfully at 03:00 UTC. Backup size: 42.8 GB. Checksum verified."),
    ("Python community meetup RSVP confirmation", "Hi {name},\nYou're registered for the upcoming Python & AI meetup next Thursday at 6:30 PM. We have talks on LangChain, scikit-learn pipelines, and MLOps."),
    ("Expense reimbursement approved", "Hi {name},\nYour expense report #{exp_num} for ${amount} has been approved and will be reimbursed in your next payroll cycle.\nHR Operations"),
    ("Office holiday schedule announcement", "All employees:\nPlease note the upcoming office closure dates for the federal holiday. Remote support shifts will be maintained as scheduled."),
]

# Templates for spam emails (SPAM)
SPAM_TEMPLATES = [
    ("URGENT: Claim your lottery prize of ${million} Million USD!", "CONGRATULATIONS! You have been selected as the grand winner of ${million},000,000 in our international sweepstakes! To release your fund immediately, click here and submit your banking info and SSN."),
    ("Earn $5,000 every day working 15 minutes from home!!", "Guaranteed passive income opportunity! No experience needed, zero risk. Start making thousands daily from your mobile phone. Click the link now to claim your starter kit before slots fill up!"),
    ("Your Bank Account Has Been Suspended - Action Required Immediately", "Dear Customer,\nUnusual debit activity was detected on your card. Your account is temporarily frozen. Verify your password, credit card number, and CVV immediately at http://secure-verify-bank-update.xyz to restore access."),
    ("Weight loss miracle - Lose 20 lbs in 7 days without exercise!", "Doctors are shocked! Discover the secret organic keto pill that burns fat overnight while you sleep. 80% discount for first 100 callers today! Buy 1 get 2 free!"),
    ("Exclusive Crypto Airdrop: Claim 10,000 Free USDT Tokens NOW!", "Limited time crypto giveaway! Connect your Web3 MetaMask wallet instantly to claim 10,000 tokens before the liquidity pool closes. Hurry, offer expires in 2 hours!"),
    ("Cheap Rolex, Gucci and Luxury Watches at 90% off!", "Top designer replica watches, bags, and luxury goods direct from factory! High quality Swiss movement, waterproof, genuine look. Order today for free discreet worldwide shipping!"),
    ("Final Notice: Your Tax Refund is Waiting to be Deposited", "Internal Revenue Alert: You have an unclaimed tax refund of $3,450.00. Click here to confirm identity and route deposit directly to your checking account: http://irs-tax-refund-portal.net"),
    ("Viagra / Cialis online pharmacy - No prescription needed!", "Best price generic medications, fast overnight shipping, discrete packaging. 100% money back guarantee. Click here for super discount pills."),
    ("Nigerian Prince Inheritance Transfer: $15.5 Million", "Dear Respected Friend,\nI am Prince Adeyemi of the royal treasury. I need a trustworthy foreign partner to assist in moving $15,500,000 USD out of the country. You will receive 35% commission. Respond with your passport copy."),
    ("YOU WON a free $500 Amazon Gift Card! Click to redeem!", "Surprise customer appreciation reward! You have won an Amazon e-gift card worth $500. Click here to answer 3 simple survey questions and redeem your code instantly!"),
    ("Urgent Wire Transfer: Outstanding Invoice #{fake_inv}", "Attention Accounts Payable:\nPlease redirect pending payments to our new beneficiary banking details attached. Urgent response required today to avoid 15% interest penalty."),
    ("Pre-Approved for $50,000 Unsecured Personal Loan - Bad Credit OK!", "Need cash fast? Instant approval loans up to $50,000 with 0% introductory interest rate. No credit check required! Apply in 60 seconds online."),
    ("Boost your website ranking to #1 on Google in 48 hours!!", "Dear Webmaster,\nWe noticed your site lacks high-PR backlinks. For only $99, we will blast 50,000 automated backlinks and rank you #1 on Google guaranteed!"),
    ("Your Netflix account is locked! Update payment information", "Dear Subscriber,\nWe could not process your latest monthly subscription fee. Your streaming service has been paused. Update your billing credentials here: http://netflix-member-billing-fix.cc"),
    ("Hot Singles In Your Area Want to Chat Tonight!", "3 matches near your zipcode are looking to connect right now! Click here to view their private uncensored photos and start live video chat."),
    ("Work at Home - Secret Shopper Wanted, Make $2000/week", "We are hiring mystery shoppers in your zip code. We send you cashier checks, you cash them and wire the remainder. Easy extra cash every week!"),
    ("Urgent: Verify Your Email Quota is 99% Full", "IT Helpdesk Warning: Your webmail storage quota has reached 99.4%. Incoming emails are being rejected. Click to validate your login credentials and expand inbox storage to 50 GB."),
    ("Invest in Green Energy Penny Stock - 500% Gains Expected", "Stock Alert: Huge announcement coming this Friday for ticker XYZ. Insiders predicting massive breakout. Buy now before market opens on Monday!"),
    ("Free Casino Bonus: $2,000 Match + 200 Free Spins", "Experience Las Vegas excitement from your home! Deposit $20 and play with $200. Guaranteed jackpots, fast crypto withdrawals!"),
    ("Notice of Appearance in Court - Case #{case_num}", "You are commanded to appear before the federal court on charges of copyright infringement. Download and extract the attached ZIP/SCR file to read your court subpoena."),
]

NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "Grace", "Henry", "Isabella", "Jack", "Karen", "Liam", "Michael", "Nina", "Oliver", "Peter", "Rachel", "Sarah", "Thomas", "Victoria"]
TIMES = ["Monday at 10:00 AM", "Tuesday at 2:30 PM", "Wednesday at 11:15 AM", "Thursday at 4:00 PM", "Friday at 9:00 AM"]
DATES = ["October 15", "November 12", "December 3", "January 18", "March 24", "April 5"]


def generate_dataset(n_samples=600, output_path="spam_ham_dataset.csv"):
    rows = []
    
    # 50% Ham, 50% Spam
    n_ham = n_samples // 2
    n_spam = n_samples - n_ham
    
    for _ in range(n_ham):
        template_subj, template_body = random.choice(HAM_TEMPLATES)
        subj = template_subj.format(
            time=random.choice(TIMES),
            pr_num=random.randint(101, 999),
            acc_num=random.randint(100000, 999999),
            amount=f"{random.uniform(25.0, 1500.0):.2f}",
            name=random.choice(NAMES),
            inv_num=random.randint(1000, 8999),
            ip_suffix=random.randint(10, 240),
            booking_ref=f"BK{random.randint(10000, 99999)}",
            date=random.choice(DATES),
            pkg_num=f"PKG{random.randint(1000000, 9999999)}",
            exp_num=random.randint(3000, 7000),
        )
        body = template_body.format(
            time=random.choice(TIMES),
            pr_num=random.randint(101, 999),
            acc_num=random.randint(100000, 999999),
            amount=f"{random.uniform(25.0, 1500.0):.2f}",
            name=random.choice(NAMES),
            inv_num=random.randint(1000, 8999),
            ip_suffix=random.randint(10, 240),
            booking_ref=f"BK{random.randint(10000, 99999)}",
            date=random.choice(DATES),
            pkg_num=f"PKG{random.randint(1000000, 9999999)}",
            exp_num=random.randint(3000, 7000),
        )
        full_text = f"Subject: {subj}\n\n{body}"
        rows.append({"label": "ham", "is_spam": 0, "subject": subj, "body": body, "text": full_text})
        
    for _ in range(n_spam):
        template_subj, template_body = random.choice(SPAM_TEMPLATES)
        subj = template_subj.format(
            million=random.randint(5, 50),
            fake_inv=random.randint(70000, 99999),
            case_num=random.randint(10000, 49999),
        )
        body = template_body.format(
            million=random.randint(5, 50),
            fake_inv=random.randint(70000, 99999),
            case_num=random.randint(10000, 49999),
        )
        full_text = f"Subject: {subj}\n\n{body}"
        rows.append({"label": "spam", "is_spam": 1, "subject": subj, "body": body, "text": full_text})
        
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} email records ({sum(df['is_spam'] == 0)} ham, {sum(df['is_spam'] == 1)} spam) -> {output_path}")
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "spam_ham_dataset.csv")
    generate_dataset(n_samples=800, output_path=output_file)
