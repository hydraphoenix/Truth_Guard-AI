"""
TruthGuard AI - Sample Content for Testing

This module provides sample real and fake news content for testing the 
misinformation detection system.
"""

# Sample real news articles
REAL_NEWS_SAMPLES = [
    {
        "id": "real_001",
        "title": "Stanford Researchers Develop New COVID-19 Treatment",
        "author": "Dr. Sarah Johnson",
        "content": """
        Scientists at Stanford University have developed a new treatment for COVID-19 
        that shows promising results in early clinical trials. The treatment, based on 
        monoclonal antibodies, has shown a 67% reduction in hospitalization rates among 
        high-risk patients. The research team, led by Dr. Sarah Johnson, published their 
        findings in the New England Journal of Medicine after a 6-month study involving 
        2,000 participants across 15 medical centers. The Food and Drug Administration 
        is currently reviewing the treatment for emergency use authorization.
        """,
        "expected_prediction": "Real",
        "confidence_threshold": 0.8,
        "source": "Legitimate medical research"
    },
    {
        "id": "real_002", 
        "title": "Tech Company Announces AI Ethics Initiative",
        "author": "Technology Reporter",
        "content": """
        A major technology company announced today that it will be investing $2 billion 
        in artificial intelligence ethics research over the next five years. The initiative 
        will focus on developing AI solutions that are fair, transparent, and beneficial 
        to society. The company's CEO stated that this investment represents their 
        commitment to responsible AI development. The program will include partnerships 
        with universities, research institutions, and civil rights organizations to 
        ensure diverse perspectives are incorporated into AI system design.
        """,
        "expected_prediction": "Real",
        "confidence_threshold": 0.7,
        "source": "Corporate announcement"
    },
    {
        "id": "real_003",
        "title": "Climate Study Shows Renewable Energy Progress", 
        "author": "Environmental Science Team",
        "content": """
        A comprehensive study published in Nature Climate Change shows that renewable 
        energy adoption has accelerated significantly over the past decade. The research, 
        conducted by an international team of climate scientists, analyzed data from 
        195 countries and found that solar and wind power now account for 12% of global 
        electricity generation. The study's lead author, Dr. Maria Rodriguez from the 
        International Energy Agency, noted that this represents a 300% increase from 
        2010 levels. The findings suggest that global renewable energy targets may be 
        achievable ahead of schedule.
        """,
        "expected_prediction": "Real",
        "confidence_threshold": 0.85,
        "source": "Scientific publication"
    }
]

# Sample fake news articles
FAKE_NEWS_SAMPLES = [
    {
        "id": "fake_001",
        "title": "SHOCKING: 5G Towers Control Your Mind!",
        "author": "Anonymous Whistleblower",
        "content": """
        URGENT ALERT!!! Secret government documents EXPOSED revealing that 5G towers 
        are actually mind control devices designed to control the population! Scientists 
        are TERRIFIED to speak out but one brave researcher has leaked the TRUTH! The 
        radiation from these towers can alter your DNA and make you obey government 
        commands! Big Tech and the government are working together to hide this from 
        you! Share this before it gets DELETED! Mainstream media won't report this 
        because they're part of the conspiracy! Wake up people!
        """,
        "expected_prediction": "Fake",
        "confidence_threshold": 0.95,
        "source": "Conspiracy theory"
    },
    {
        "id": "fake_002",
        "title": "Doctors HATE This One Weird Trick",
        "author": "Health Guru Mike",
        "content": """
        Medical professionals HATE this one weird trick that cures EVERYTHING! A local 
        mom discovered this ancient secret that Big Pharma has been hiding for decades. 
        Just drink this special water mixture every morning and you'll never need 
        medicine again! This miracle cure has been suppressed by the medical establishment 
        because it would put them out of business! Doctors are losing their minds over 
        this discovery! Pharmaceutical companies are trying to ban this information! 
        Click here to learn the secret they don't want you to know!
        """,
        "expected_prediction": "Fake", 
        "confidence_threshold": 0.98,
        "source": "Health misinformation"
    },
    {
        "id": "fake_003",
        "title": "BREAKING: Aliens Land in Nevada Desert",
        "author": "UFO Truth Seeker",
        "content": """
        EXCLUSIVE FOOTAGE shows alien spacecraft landing near Area 51! Government 
        sources confirm that extraterrestrial contact has been made and the aliens 
        have been negotiating with world leaders for months. This is being covered 
        up by mainstream media and government agencies! The aliens revealed advanced 
        technology that could solve all of humanity's problems, but the government 
        wants to keep it secret! Only independent journalists are reporting the TRUTH! 
        Share this before the government shuts down the internet! They're already 
        censoring this information on social media!
        """,
        "expected_prediction": "Fake",
        "confidence_threshold": 0.92,
        "source": "UFO conspiracy"
    }
]

# Edge case test content
EDGE_CASES = {
    "very_short": {
        "content": "Breaking news!",
        "description": "Very short content",
        "expected_behavior": "Should handle gracefully or request more content"
    },
    "very_long": {
        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 500,
        "description": "Very long content (500 repetitions)",
        "expected_behavior": "Should process within time limits"
    },
    "special_characters": {
        "content": "Test with émojis 🚨, sp3c!@l ch@rs, and FORMATTING issues...",
        "description": "Special characters and formatting",
        "expected_behavior": "Should handle Unicode and special characters"
    },
    "empty_string": {
        "content": "",
        "description": "Empty string",
        "expected_behavior": "Should return appropriate error message"
    },
    "whitespace_only": {
        "content": "   \n\t   ",
        "description": "Whitespace only",
        "expected_behavior": "Should treat as empty content"
    },
    "numbers_only": {
        "content": "12345 67890 2024 99.9% 3.14159",
        "description": "Numbers and symbols only",
        "expected_behavior": "Should handle non-textual content"
    },
    "html_content": {
        "content": "<script>alert('test')</script>This is a test article with HTML tags.",
        "description": "Content with HTML/script tags",
        "expected_behavior": "Should sanitize and process safely"
    },
    "urls_and_emails": {
        "content": "Visit https://example.com or email test@example.com for more information.",
        "description": "Content with URLs and emails",
        "expected_behavior": "Should handle URLs and email addresses appropriately"
    }
}

# Mixed/satirical content that could be challenging
CHALLENGING_CASES = [
    {
        "id": "satirical_001",
        "title": "Local Man Discovers Revolutionary Way to Stay Hydrated",
        "content": """
        In a groundbreaking discovery that has scientists baffled, local resident Jim 
        Thompson has reportedly found that drinking water helps him stay hydrated. This 
        revolutionary finding challenges decades of conventional wisdom and has researchers 
        scrambling to understand the implications. 'I was skeptical at first,' said Dr. 
        Maria Rodriguez, a hydration expert, 'but our extensive testing confirms that 
        H2O does indeed prevent dehydration.' Thompson plans to patent his discovery and 
        market it as 'Hydration Solution 1.0.'
        """,
        "expected_prediction": "Real",  # Satirical but factually accurate
        "confidence_threshold": 0.7,
        "source": "Satirical news (factually correct)",
        "note": "Tests handling of satirical content"
    },
    {
        "id": "opinion_001",
        "title": "Why Pineapple Doesn't Belong on Pizza",
        "content": """
        The great pineapple pizza debate has raged for decades, but it's time to settle 
        this once and for all. Pineapple simply does not belong on pizza. The sweetness 
        of the fruit clashes with the savory elements of cheese and tomato sauce, creating 
        a culinary catastrophe. Real pizza purists understand that traditional Italian 
        recipes never included tropical fruits. This isn't about being closed-minded; 
        it's about respecting the integrity of one of the world's greatest foods.
        """,
        "expected_prediction": "Real",  # Opinion piece, not misinformation
        "confidence_threshold": 0.6,
        "source": "Opinion article",
        "note": "Tests handling of opinion vs. factual content"
    }
]

# Test data for performance testing
PERFORMANCE_TEST_DATA = {
    "small_batch": REAL_NEWS_SAMPLES[:1] + FAKE_NEWS_SAMPLES[:1],
    "medium_batch": REAL_NEWS_SAMPLES + FAKE_NEWS_SAMPLES,
    "large_batch": (REAL_NEWS_SAMPLES + FAKE_NEWS_SAMPLES) * 10,
    "stress_test": (REAL_NEWS_SAMPLES + FAKE_NEWS_SAMPLES) * 50
}

# Helper functions
def get_real_news_samples():
    """Get all real news samples"""
    return REAL_NEWS_SAMPLES

def get_fake_news_samples():
    """Get all fake news samples"""
    return FAKE_NEWS_SAMPLES

def get_all_samples():
    """Get all sample content"""
    return REAL_NEWS_SAMPLES + FAKE_NEWS_SAMPLES

def get_edge_cases():
    """Get edge case test content"""
    return EDGE_CASES

def get_challenging_cases():
    """Get challenging test cases"""
    return CHALLENGING_CASES

def get_sample_by_id(sample_id):
    """Get specific sample by ID"""
    all_samples = get_all_samples() + get_challenging_cases()
    for sample in all_samples:
        if sample.get('id') == sample_id:
            return sample
    return None

def get_performance_batch(size="small"):
    """Get batch of samples for performance testing"""
    return PERFORMANCE_TEST_DATA.get(f"{size}_batch", PERFORMANCE_TEST_DATA["small_batch"])

# Expected performance metrics
EXPECTED_PERFORMANCE = {
    "processing_time": {
        "max_seconds": 2.0,
        "target_seconds": 1.0
    },
    "accuracy": {
        "min_accuracy": 0.90,
        "target_accuracy": 0.95
    },
    "memory": {
        "max_mb": 2048,
        "target_mb": 1024
    }
}

if __name__ == "__main__":
    # Test the sample data
    print("TruthGuard AI - Sample Content")
    print("=" * 40)
    print(f"Real news samples: {len(REAL_NEWS_SAMPLES)}")
    print(f"Fake news samples: {len(FAKE_NEWS_SAMPLES)}")
    print(f"Edge cases: {len(EDGE_CASES)}")
    print(f"Challenging cases: {len(CHALLENGING_CASES)}")
    print(f"Total test samples: {len(get_all_samples()) + len(CHALLENGING_CASES)}")