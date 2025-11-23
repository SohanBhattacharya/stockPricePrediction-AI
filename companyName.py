class companyName:
    company_tickers = {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Amazon.com, Inc.": "AMZN",
        "Alphabet Inc. (Class A)": "GOOGL",
        "Tesla, Inc.": "TSLA",
        "JPMorgan Chase & Co.": "JPM",
        "Walmart Inc.": "WMT",
        "Procter & Gamble Company": "PG",
        "Exxon Mobil Corporation": "XOM",
        "Meta Platforms, Inc.": "META",
        "Intel Corporation": "INTC",
        "The Coca-Cola Company": "KO",
        "PepsiCo, Inc.": "PEP",
        "Pfizer Inc.": "PFE",
        "Chevron Corporation": "CVX",
        "Netflix, Inc.": "NFLX",
        "Adobe Inc.": "ADBE",
        "Salesforce, Inc.": "CRM",
        "International Business Machines Corporation": "IBM",
        "The Walt Disney Company": "DIS",
        "McDonald’s Corporation": "MCD",
        "Nike, Inc.": "NKE",
        "Cisco Systems, Inc.": "CSCO",
        "Oracle Corporation": "ORCL",
        "AbbVie Inc.": "ABBV",
        "Merck & Co., Inc.": "MRK",
        "Infosys Ltd.": "INFY"
    }

    # (Your PRODUCT_PORTFOLIOS dictionary — unchanged except we may patch GOOG below)
    PRODUCT_PORTFOLIOS = {
        "AAPL": {
            "Products": {
                "iPhone": ["iPhone 14", "iPhone 15", "iPhone SE"],
                "Mac": ["Macbook Air", "Macbook Pro", "iMac", "Mac Mini"],
                "iPad": ["iPad Pro", "iPad Air", "iPad Mini"]
            },
            "Services": {
                "Subscriptions": ["iCloud", "Apple Music", "Apple TV+", "Apple Arcade"],
                "Payments": ["Apple Pay", "Apple Card"]
            }
        },
        "MSFT": {
            "Software": {
                "Windows": ["Win 10", "Win 11"],
                "Microsoft Office": ["Word", "Excel", "PowerPoint"]
            },
            "Cloud & Business": {
                "Azure": ["AI Services", "Compute", "Storage"],
                "LinkedIn": ["Premium", "Learning"]
            },
            "Gaming": {
                "Xbox": ["Games", "Console", "GamePass"]
            }
        },
        "AMZN": {
            "E-Commerce": {
                "Retail": ["Amazon Basics", "Fashion", "Electronics"],
                "Marketplaces": ["Amazon Sellers", "Third-Party Seller Brands"]
            },
            "Cloud": {
                "AWS": ["Compute", "AI & ML", "Storage", "Security"]
            },
            "Media": {
                "Prime Video": ["Streaming", "Original Programming"],
                "Audible": ["Audiobooks", "Podcasts"]
            }
        },
        "GOOGL": {
            "Advertising": {
                "Google Ads": ["Search Ads", "Display Ads"],
                "YouTube": ["Subscriptions", "Ads"]
            },
            "Software": {
                "Android": ["OS", "Play Store"],
                "Chrome": ["Browser", "Chromebook Ecosystem"]
            },
            "Cloud": {
                "Google Cloud": ["Compute", "AI", "Data Analytics"]
            }
        },
        "GOOGL": {
            "Advertising": {
                "Google Ads": ["Search Ads", "Display Ads"],
                "YouTube": ["Subscriptions", "Ads"]
            },
            "Software": {
                "Android": ["OS", "Play Store"],
                "Chrome": ["Browser", "Chromebook Ecosystem"]
            },
            "Cloud": {
                "Google Cloud": ["Compute", "AI", "Data Analytics"]
            }
        },
        "TSLA": {
            "Automotive": {
                "Sedans": ["Model 3", "Model S"],
                "SUVs": ["Model Y", "Model X"],
                "Cybertruck": []
            },
            "Energy": {
                "Solar": ["Panels", "Roof"],
                "Storage": ["Powerwall", "Megapack"]
            },
            "Software": {
                "Autopilot": ["FSD", "Navigation", "OTA Updates"]
            }
        },
        "JPM": {
            "Banking": {
                "Retail": ["Savings", "Checking", "Credit"],
                "Commercial": ["Loans", "Investments"]
            },
            "Wealth": {
                "Asset Management": ["ETFs", "Advisory"]
            }
        },
        "WMT": {
            "Retail": {
                "Stores": ["Supercenters", "Neighborhood Markets"],
                "Online": ["Walmart.com", "Marketplace"]
            },
            "Services": {
                "Financial": ["Money Transfer", "Credit Cards"]
            }
        },
        "PG": {
            "Consumer Goods": {
                "Home Care": ["Tide", "Ariel", "Dawn"],
                "Personal Care": ["Gillette", "Head & Shoulders"]
            }
        },
        "XOM": {
            "Energy": {
                "Oil": ["Exploration", "Refining"],
                "Gas": ["LNG", "Natural Gas"]
            },
            "Chemicals": ["Plastics", "Packaging Materials"]
        },
        "META": {
            "Social": {
                "Platforms": ["Facebook", "Instagram", "Threads"]
            },
            "VR & Metaverse": {
                "Meta Quest": ["Quest 2", "Quest 3"]
            },
            "Messaging": ["WhatsApp", "Messenger"]
        },
        "INTC": {
            "Chips": {
                "Consumer": ["Laptop CPUs", "Desktop CPUs"],
                "Enterprise": ["Xeon", "Server Chips"]
            },
            "Software": ["Intel AI Stack"]
        },
        "KO": {
            "Beverages": {
                "Soft Drinks": ["Coca-Cola", "Sprite", "Fanta"],
                "Water": ["Dasani", "Smartwater"],
                "Juices": ["Minute Maid"]
            }
        },
        "PEP": {
            "Food & Snacks": ["Lays", "Doritos", "Cheetos"],
            "Beverages": ["Pepsi", "Mountain Dew", "Gatorade"]
        },
        "PFE": {
            "Pharmaceuticals": ["Vaccines", "Oncology", "Virology"]
        },
        "CVX": {
            "Energy": ["Oil Production", "Gas", "Petrochemicals"],
            "Renewables": ["Biofuels", "Hydrogen"]
        },
        "NFLX": {
            "Streaming": {
                "Content": ["Movies", "Series"],
                "Plans": ["Standard", "Premium", "Ads Tier"]
            }
        },
        "ADBE": {
            "Software": {
                "Creative Cloud": ["Photoshop", "Illustrator", "Premiere Pro"],
                "Document Cloud": ["Acrobat", "PDF Services"]
            }
        },
        "CRM": {
            "Software": {
                "Salesforce Cloud": ["Sales", "Service", "Marketing"],
                "AI": ["Einstein AI"]
            }
        },
        "IBM": {
            "Cloud": ["Hybrid Cloud", "AI Infrastructure"],
            "Enterprise": ["Consulting", "Quantum Computing"]
        },
        "DIS": {
            "Entertainment": ["Disney+", "Hulu", "ESPN"],
            "Parks": ["Disneyland", "Disney World"],
            "Studios": ["Marvel", "Pixar", "Lucasfilm"]
        },
        "MCD": {
            "Food": {
                "Menu": ["Burgers", "McCafe", "Fries"],
                "Delivery": ["McDelivery", "Drive-Thru"]
            }
        },
        "NKE": {
            "Products": {
                "Footwear": ["Air Max", "Jordan", "Running"],
                "Apparel": ["Sportswear", "Athletics"]
            }
        },
        "CSCO": {
            "Networking": ["Routers", "Switches"],
            "Security": ["Firewalls", "Threat Detection"]
        },
        "ORCL": {
            "Software": ["Databases", "Oracle Cloud"],
            "Enterprise": ["ERP", "HR Systems"]
        },
        "ABBV": {
            "Pharmaceuticals": ["Immunology", "Oncology"]
        },
        "MRK": {
            "Healthcare": ["Vaccines", "Oncology", "Animal Health"]
        },
        "INFY": {
            "IT Services": ["Consulting", "Cloud", "Cybersecurity"],
            "Platforms": ["Finacle", "AI Tools"]
        }
    }
