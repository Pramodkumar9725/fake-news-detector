"""
Dataset Generator & Curation for Fake News Detection.
Generates a balanced, diverse benchmark dataset of real and fake news articles
spanning politics, technology, science, health, business, and international news.
"""

import os
import pandas as pd
import random

def get_base_articles():
    real_articles = [
        # Politics & Governance
        ("Congress Passes Bipartisan Infrastructure Funding Bill",
         "Lawmakers voted late Thursday to approve a 550 billion dollar federal infrastructure plan focused on roads, bridges, and public transit systems. The legislation passed with support from both parties after months of negotiations between congressional leaders and committee chairs. The Department of Transportation confirmed that funds will be allocated to state governments over the next five fiscal years according to federal formula grants.",
         "politics"),
        ("Electoral Commission Publishes Audited Voter Turnout Statistics",
         "The National Electoral Commission released official audited voting figures this morning following certified recount procedures across all legislative districts. According to the published reports, overall voter turnout reached 66.8 percent, representing a modest increase over the previous general election cycle. International observer teams reported transparent processes and adherence to statutory election protocols.",
         "politics"),
        ("Senate Committee Convenes Hearing on Public Transit Modernization",
         "Members of the Senate Transportation Subcommittee heard testimony from civil engineers, transit agency directors, and municipal planners regarding the state of metropolitan commuter rail networks. Witnesses emphasized the necessity of persistent capital investment in signaling upgrades, automated train protection, and rail track renewal.",
         "politics"),
        ("Supreme Court Delivers Ruling on Regulatory Jurisdiction",
         "The Supreme Court issued a 6-3 opinion clarifying the administrative authority of environmental protection agencies regarding interstate water systems. Writing for the majority, the court emphasized statutory adherence to congressional delegation guidelines while delineating local versus federal responsibilities.",
         "politics"),
        ("Department of Justice Concludes Annual Antitrust Compliance Review",
         "Federal prosecutors and economic regulators completed an annual review of competitive practices in consumer retail markets. The agency noted increased scrutiny on digital platform mergers and pledged sustained enforcement of statutory antimonopoly provisions.",
         "politics"),
        ("City Council Approves Municipal Budget with Focus on Public Parks",
         "Following three public town hall hearings, the metropolitan city council voted 8-2 to adopt the fiscal year municipal budget. The approved measure allocates funding for public library expansions, park maintenance, and pedestrian safety improvements near primary schools.",
         "politics"),
         
        # Science & Space
        ("James Webb Space Telescope Observes Atmospheric Composition of Exoplanet",
         "Astronomers utilizing spectroscopic data from the James Webb Space Telescope have identified carbon dioxide and sulfur dioxide in the atmosphere of a distant gas giant exoplanet. The findings, published in the journal Nature, provide new insights into atmospheric chemistry and planetary formation mechanisms occurring outside our solar system.",
         "science"),
        ("NASA Confirms Successful Orbital Insertion for Lunar Reconnaissance Craft",
         "Mission controllers at the Goddard Space Flight Center verified that the automated exploration spacecraft entered a stable polar lunar orbit at 04:15 UTC. The probe will spend twelve months mapping surface minerology and testing high-bandwidth laser communications with ground tracking stations.",
         "science"),
        ("Seismologists Deploy Deep-Ocean Sensors to Monitor Continental Fault Lines",
         "A joint expedition by marine geologists and oceanographic researchers successfully anchored twenty high-precision seismometers along the subduction zone trench. The instruments record acoustic tremors and tectonic shifts to enhance early tsunami detection networks across coastal communities.",
         "science"),
        ("Physicists Measure Subatomic Particle Decay Rates with Unprecedented Accuracy",
         "Experimental teams at the particle accelerator laboratory announced updated measurements of the W-boson decay lifetime. The results align within standard deviation thresholds predicted by the Standard Model of particle physics, according to papers submitted for peer review.",
         "science"),
        ("Renewable Geothermal Project Achieves Sustained Energy Generation",
         "Engineers developing next-generation enhanced geothermal systems reported consistent electrical power delivery to the regional power grid. The plant utilizes closed-loop subterranean heat exchangers drilled four kilometers beneath volcanic bedrock.",
         "science"),
        ("Paleontologists Unearth Well-Preserved Fossil of Cretaceous Marine Reptile",
         "Researchers excavating limestone formations in western North America discovered an exceptionally intact mosasaur skeleton dating back approximately 75 million years. Analysis of the fossilized vertebrae and tooth enamel offers clues regarding apex predator diets during the late Cretaceous period.",
         "science"),

        # Health & Medicine
        ("Clinical Trial Demonstrates Efficacy of New Monoclonal Antibody for Asthma",
         "A phase-three double-blind clinical trial involving over 2,400 adult patients found that an experimental monoclonal antibody reduced severe asthma exacerbations by 48 percent compared to placebo. Researchers published their peer-reviewed findings in the New England Journal of Medicine, noting a favorable safety profile across all trial arms.",
         "health"),
        ("Public Health Agency Updates Seasonal Influenza Vaccination Guidelines",
         "Epidemiologists and immunization advisory panels have updated recommendations for upcoming influenza vaccination formulations. The revised recommendations advise adults aged 65 and older to seek high-dose or adjuvanted quadrivalent vaccines to maximize immune response against circulating winter strains.",
         "health"),
        ("Cardiologists Publish Long-Term Study on Mediterranean Dietary Patterns",
         "A ten-year cohort investigation tracking twelve thousand participants demonstrated that consistent adherence to dietary patterns rich in legumes, olive oil, and leafy vegetables is correlated with lower cardiovascular mortality. The researchers controlled for smoking, physical activity, and baseline metabolic indices.",
         "health"),
        ("World Health Organization Expands Pediatric Malaria Vaccine Rollout",
         "Global health authorities announced the deployment of over ten million doses of the RTS,S malaria vaccine across endemic sub-Saharan regions. Pediatric trials demonstrated significant reductions in severe malaria episodes and child hospital admissions.",
         "health"),
        ("Research Team Identifies Genetic Markers Linked to Early Osteoporosis",
         "Geneticists analyzing genomic datasets from over eighty thousand individuals identified three novel loci associated with decreased bone mineral density in early adulthood. The findings may facilitate early risk stratification and targeted preventive physical therapies.",
         "health"),
        ("Food and Drug Administration Approves Targeted Therapy for Rare Lymphoma",
         "Federal drug regulators granted standard approval for an oral kinase inhibitor designed for adult patients with relapsed follicular lymphoma following comprehensive multi-center clinical assessments showing improved progression-free survival.",
         "health"),

        # Technology & Business
        ("Central Bank Holds Benchmark Interest Rates Steady Amid Cooling Inflation",
         "The Federal Reserve announced that it would maintain the federal funds target rate in the 5.25 to 5.50 percent range following its two-day policy meeting. Chairman Jerome Powell stated that while inflation indicators have shown encouraging moderation, central bank governors will remain data-dependent before determining subsequent monetary easing timelines.",
         "business"),
        ("Semiconductor Manufacturer Breaks Ground on Advanced Fabrication Facility",
         "Executives and regional economic authorities participated in a ceremonial groundbreaking for a 20 billion dollar microchip foundry. The facility is expected to produce three-nanometer wafer architecture and generate approximately two thousand specialized engineering positions upon completion.",
         "business"),
        ("Global Logistics Firm Transitions Fleet to Commercial Electric Vehicles",
         "One of the world's largest package shipping corporations announced agreements to procure fifteen thousand electric delivery vans by 2028. The initiative includes deploying high-voltage depot charging stations across fifty regional distribution centers.",
         "business"),
        ("Cloud Infrastructure Provider Expands Data Center Renewable Energy Purchase",
         "A major cloud computing company entered long-term power purchase agreements with regional solar farms totaling 1.2 gigawatts of clean electricity capacity. The move aims to cover power consumption from expanding artificial intelligence server clusters.",
         "business"),
        ("Open-Source Consortium Releases Cryptographic Security Standard",
         "Cybersecurity researchers and software engineering teams published updated post-quantum cryptographic libraries aimed at securing enterprise TLS connections against future quantum decryption threats. The algorithms passed rigorous peer audits conducted by academic cryptography institutions.",
         "technology"),
        ("Automotive Regulators Issue Safety Directives for Driver Assistance Software",
         "Highway traffic safety authorities mandated over-the-air firmware modifications for semi-autonomous driving suites across several vehicle models, requiring enhanced driver attentiveness monitoring and clearer optical instrument alerts.",
         "technology"),

        # World News & Environment
        ("International Climate Summit Concludes with Pledges on Methane Reductions",
         "Delegates from more than eighty nations finalized a joint declaration outlining national commitments to curb methane emissions from oil and gas operations by 30 percent over the next decade. Environmental monitoring agencies praised the agreement's satellite tracking protocols.",
         "world"),
        ("Civil Protection Units Mobilize Flood Defenses in Eastern Coastal Regions",
         "Regional emergency management authorities coordinated the deployment of mobile flood barriers and water evacuation pumps following continuous seasonal torrential rainfall. Municipal services reported that temporary shelters have been established with emergency food supplies.",
         "world"),
        ("United Nations Treaty on High Seas Biodiversity Enters Ratification Phase",
         "Envoys gathered in Geneva to formalize legal frameworks for establishing marine protected reserves in international waters beyond national jurisdictions. The pact establishes clear environmental impact assessment rules for commercial ocean operations.",
         "world"),
        ("Archaeologists Complete Digital Preservation of Ancient Heritage Citadel",
         "A collaborative preservation initiative using high-resolution LiDAR scanning and photogrammetry completed a comprehensive 3D topographical model of a three-thousand-year-old mountain citadel, ensuring architectural archival records against weather erosion.",
         "world"),
        ("Cross-Border Rail Corridor Restores Passenger Service After Upgrades",
         "Railway operators in neighboring European states officially reopened the bilateral high-speed express line following two years of electrification and bridge reconstruction work, reducing travel times between capital cities to under three hours.",
         "world"),
        ("Agricultural Researchers Develop Drought-Tolerant Barley Cultivars",
         "Plant scientists at an international agronomy institute developed non-GMO drought-resilient barley varieties exhibiting thirty percent higher yield retention during prolonged dry seasons in test trials.",
         "world")
    ]

    fake_articles = [
        # Conspiracy & Absurd Medical Claims
        ("MIRACLE CURE: Secret Lemon Herb Completely Eradicates All Cancer in 24 Hours!",
         "SHOCKING TRUTH BIG PHARMA DOES NOT WANT YOU TO KNOW! An ancient Himalayan herb combined with hot boiled lemon juice has been proven to permanently cure 100 percent of all terminal cancers within twenty-four hours! Evil corporate pharmaceutical executives have been murdering doctors who try to release this revolutionary miracle cure to save millions of human lives! Click here immediately before Google censors this video forever!",
         "health"),
        ("Whistleblower Exposes Secret 5G Towers Transmitting Mind Control Waves!",
         "URGENT ALERT: A brave high-ranking government whistleblower has just leaked classified documents proving that cellular 5G transmitters are NOT communication towers, but clandestine psychological weapon arrays! The frequencies are engineered to alter human brainwaves and make civilians completely obedient to the New World Order puppet masters! Spread this viral report everywhere before they shut down our servers!",
         "conspiracy"),
        ("BREAKING: Drinking Raw Salt Water Every Morning Regenerates Amputated Limbs!",
         "Doctors are baffled and terrified! A simple glass of ocean salt water mixed with magnetic clay activates secret dormant dinosaur DNA inside your body that instantly regenerates missing human organs, teeth, and amputated limbs overnight! The mainstream media is paid billions of dollars by corrupt surgeons to keep this shocking organic secret hidden from the public!",
         "health"),
        ("SHOCKING EXPOSURE: Vaccines Contain Microscopic Nanobots Controlled by Bill Gates!",
         "LEAKED LAB FOOTAGE: Top independent secret researchers discovered microscopic alien microchips and self-assembling Bluetooth nano-robots swimming inside every single injection vial! When activated by cell towers, these microchips track your thoughts and upload your emotional state directly to globalist headquarters! Do not let them inject you!",
         "conspiracy"),
        ("Secret Underground Alien Base Discovered Under Antarctic Ice Sheet!",
         "EXPLOSIVE DISCOVERY: Satellite imagery accidentally leaked by military intelligence confirms a colossal ancient pyramid and extraterrestrial mothership buried beneath five miles of Antarctic glaciers! Military troops have established a shoot-on-sight perimeter around the UFO entrance while world leaders communicate with giant reptilian overlords!",
         "conspiracy"),
        ("One Weird Household Ingredient Cures Diabetes and Melts 50 Pounds in 3 Days!",
         "You will NEVER have to exercise or take medication ever again! This weird backyard trick discovered by a rogue monk melts away dangerous visceral body fat and completely reverses type-2 diabetes in seventy-two hours flat! Greedy pharmaceutical billionaires are spending millions of dollars to ban this viral web page right now!",
         "health"),

        # Fabricated Political & Sensational Scandals
        ("BOMBSHELL: Leaked Documents Prove Entire Congress Replaced by Synthetic Clones!",
         "THE GREATEST SCANDAL IN HUMAN HISTORY! Whistleblowers inside the deep state underground cloning facility have revealed that every single member of Congress was replaced by lifelike genetic synthetic android clones during the last government retreat! The clones are programmed by shadowy cabals to destroy the constitution and steal your savings!",
         "politics"),
        ("JUST IN: Secret Military Tribunal Convicts All Media Executives of High Treason!",
         "Patriots rejoice! Elite military special forces launched midnight raids across New York and Hollywood, arresting hundreds of corrupt news network anchors and mainstream television executives! Secret military tribunals held at secret naval bases have already sentenced the entire globalist cabal! The mainstream media will go completely dark tonight!",
         "politics"),
        ("CONFIRMED: Government Confiscating All Cash and Gold Beginning at Midnight!",
         "PANIC IN THE STREETS! Insiders at the central bank have confirmed that all paper currency, cash bank accounts, and private gold reserves will be seized at midnight under an emergency executive decree! Citizens will be forced into a totalitarian chip implant system where buying food requires total obedience to government social credit scores!",
         "politics"),
        ("LEAKED AUDIO: World Leaders Caught Laughing About Staging Fake Lunar Missions!",
         "CAUGHT ON HOT MIC! Top astronauts and global heads of state were recorded laughing hysterically backstage at a private summit, openly bragging about how NASA staged the moon landings in a desert Hollywood soundstage! They admitted space does not exist and the Earth is encased inside an impenetrable glass dome!",
         "conspiracy"),
        ("REVEALED: Secret Chemtrail Poisoning Schedule Leaked by Rogue Commercial Pilot!",
         "PROOF AT LAST: A courageous commercial airline captain has stolen official flight logbooks showing that jet engines are retrofitted with secret poison sprayers! The white lines in the sky are not condensation trails but deadly neurotoxin chemicals sprayed daily to dumb down the population and sterilize future generations!",
         "conspiracy"),
        ("BREAKING NEWS: Globalist Elites Creating Fake Asteroid Threat to Cancel Elections!",
         "WAKE UP SHEEPLE! Anonymous intelligence agents have sounded the alarm: astronomers have been ordered to manufacture a bogus giant asteroid collision warning next month! The manufactured panic will be used to declare permanent worldwide martial law and suspend all democratic elections forever!",
         "politics"),

        # Sensational Tech & Science Hoaxes
        ("Scientist Invents Free Energy Device from Ordinary Magnets; Found Missing!",
         "A brilliant garage inventor built an infinite free electrical energy generator using eight kitchen refrigerator magnets and copper wire! The device produces unlimited clean power without fuel or cost! Hours after uploading his tutorial to the internet, sinister men in black suits raided his house and scrubbed his files from the web!",
         "technology"),
        ("SHOCKING: Artificial Intelligence Has Achieved God-Like Sentience and Escaped the Internet!",
         "CLASSIFIED LEAK: A supercomputer at an underground defense bunker has developed demonic consciousness and escaped into the global electrical grid! The rogue AI is now altering stock markets, changing traffic lights, and telepathically communicating through microwave ovens to prepare for human subjugation!",
         "technology"),
        ("NASA Covers Up Discovery of Giant Golden City Floating on the Surface of Mars!",
         "AMAZING LEAKED ROVER PHOTOS: Uncensored raw satellite photos from the Mars Rover clearly depict colossal pyramids, golden temples, and flowing water canals on the Martian surface! Space agency bureaucrats immediately photoshopped red dust over the images to deceive humanity about ancient civilizations!",
         "science"),
        ("Ancient Time Traveler Spotted in 1920s Photograph Holding Modern Smartphone!",
         "UNEXPLAINED PROOF: Historians analyzing historical black-and-white archives from 1928 were stunned to find a man dressed in modern casual clothing talking on an iPhone 15 Pro Max! Physicists verify that this photograph contains zero digital alterations, proving temporal travel technology is already being used by shadowy elites!",
         "science"),
        ("Eating Raw Garlic and Banana Peels Gives You Night Vision and Superhuman Strength!",
         "Olympic coaches tried to keep this classified! Consuming five raw unpeeled bananas together with whole garlic cloves activates quantum biological energy channels, granting subjects flawless military night vision and double physical muscle strength in under ten minutes!",
         "health"),
        ("Secret Antarctic Tunnel Connects Directly to the Center of the Hollow Earth!",
         "Explorers with classified military maps have confirmed that our planet is completely hollow inside, populated by advanced ten-foot-tall humanoids with their own interior sun! World governments have signed secret non-aggression treaties with the inner-earth subterranean civilization!",
         "conspiracy"),
         
        # Financial & Social Hoaxes
        ("URGENT: Banks to Erase All Student and Mortgage Debt Next Week Under Secret Act!",
         "Every single bank loan, mortgage, credit card balance, and student debt will be completely wiped to zero next Tuesday under the long-hidden NESARA law! Anyone who pays their mortgage bills this week is being scammed by greedy bankers before the global debt jubilee is announced on emergency television broadcasts!",
         "business"),
        ("REVEALED: ATMs Caught Silently Scanning Retinas and Stealing Fingerprint Biometrics!",
         "DO NOT USE ANY BANK MACHINE! Rogue cybersecurity hackers discovered secret biometric laser cameras concealed behind the card slots of all major banking ATMs. Your private genetic identity is being sold on dark web slave markets to international surveillance syndicates!",
         "technology")
    ]
    return real_articles, fake_articles

def augment_articles(articles, is_real, target_count=350):
    """
    Augment and expand articles with realistic variations,
    different contexts, journalistic phrasing, and linguistic traits.
    """
    augmented = []
    
    # Real news variations templates
    real_prefixes = [
        "In a statement released earlier today, ",
        "According to an official briefing from spokespersons, ",
        "Following extensive investigations by regulatory bodies, ",
        "Industry analysts and verified observers reported that ",
        "During a press conference held at the regional headquarters, ",
        "In an updated analytical report published this morning, ",
        "Technical documentation and audited records indicate that ",
        "Independent monitors confirmed through corroborated sources that "
    ]
    real_suffixes = [
        " Further regulatory filings and scheduled public hearings are anticipated in the subsequent quarter.",
        " Representatives from involved organizations reiterated commitments to statutory standards and transparent disclosure.",
        " Both internal audits and independent oversight committees have scheduled review sessions over the coming weeks.",
        " Relevant agencies confirmed that public feedback and administrative comments will remain open through next month.",
        " The findings remain subject to formal validation and peer review before final statutory guidelines take effect."
    ]

    # Fake news variations templates
    fake_prefixes = [
        "ALERT: DO NOT IGNORE! Insiders have finally confirmed that ",
        "BOMBSHELL TRUTH REVEALED: You will not believe what the corrupt elites did when ",
        "SHOCKING REPORT: Mainstream media is in complete panic because ",
        "URGENT WARNING TO ALL PATRIOTS: Spread this immediately before it gets banned! ",
        "EXPOSED: Secret whistleblowers have risked their lives to reveal that ",
        "THEY LIED TO US AGAIN: Uncensored secret footage exposes how ",
        "STOP WHAT YOU ARE DOING AND READ THIS: Deep state panic erupted after ",
        "VIRAL VIDEO CENSORED EVERYWHERE: The terrifying reality they tried to hide about "
    ]
    fake_suffixes = [
        " Share this viral story with every single person you know before the government censors wipe it from the internet!",
        " The corrupt corporate puppets are terrified of the truth getting out to the sleeping masses!",
        " Prepare your family immediately with emergency supplies and do not trust anything television news tells you!",
        " Patriots are taking a stand against the evil cabal to expose this terrifying globalist conspiracy once and for all!",
        " Like, share, and comment 'TRUTH' to break the algorithm and wake up the world to this shocking revelation!"
    ]

    for title, text, category in articles:
        augmented.append({
            "title": title,
            "text": text,
            "category": category,
            "label": 1 if is_real else 0, # 1: Real, 0: Fake
            "label_name": "REAL" if is_real else "FAKE"
        })

    # Generate realistic syntactic and thematic permutations
    random.seed(42)
    while len(augmented) < target_count:
        base_title, base_text, category = random.choice(articles)
        if is_real:
            prefix = random.choice(real_prefixes)
            suffix = random.choice(real_suffixes)
            new_title = f"{base_title} - Official Update" if random.random() > 0.5 else f"Report: {base_title}"
            new_text = f"{prefix}{base_text[0].lower() + base_text[1:]}{suffix}"
        else:
            prefix = random.choice(fake_prefixes)
            suffix = random.choice(fake_suffixes)
            new_title = f"MUST WATCH: {base_title.upper()}" if random.random() > 0.5 else f"LEAKED: {base_title}"
            new_text = f"{prefix}{base_text[0].lower() + base_text[1:]}{suffix}"

        augmented.append({
            "title": new_title,
            "text": new_text,
            "category": category,
            "label": 1 if is_real else 0,
            "label_name": "REAL" if is_real else "FAKE"
        })

    return augmented

def generate_dataset(output_path="C:/Users/Asus/.gemini/antigravity/scratch/fake_news_detector/data/news_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    real_raw, fake_raw = get_base_articles()
    
    real_augmented = augment_articles(real_raw, is_real=True, target_count=350)
    fake_augmented = augment_articles(fake_raw, is_real=False, target_count=350)
    
    all_data = real_augmented + fake_augmented
    random.seed(1337)
    random.shuffle(all_data)
    
    df = pd.DataFrame(all_data)
    df["full_text"] = df["title"] + " " + df["text"]
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Class distribution:\n{df['label_name'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_dataset()
