from simple_video_detector import SimpleVideoDetector
import os

def test_video(video_path):
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize detector
    print(f"\n{'='*50}")
    print("Initializing detector...")
    detector = SimpleVideoDetector()
    
    # Analyze video
    print(f"\nAnalyzing video: {video_path}")
    print("This may take a few moments...")
    result = detector.detect_deepfake(video_path)
    
    # Check if result is valid
    if result is None or 'error' in result:
        print(f"Error analyzing video: {result.get('error') if result else 'Unknown error'}")
        return
    
    # Print results in a clear format
    print(f"\n{'='*50}")
    print("ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"VERDICT: This video is {result['result'].upper()}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nProbabilities:")
    print(f"• Fake: {result['probability_fake']}%")
    print(f"• Real: {result['probability_real']}%")
    
    print(f"\nDetailed Analysis:")
    for detail in result['analysis']['details']:
        print(f"• {detail}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # First, check video files
    videos = [
        "test_videos/real_video_1.mp4",
        "test_videos/deepfake_1.mp4"
    ]
    
    # Print current directory and check files
    print("Current directory:", os.getcwd())
    print("\nChecking video files:")
    for video_path in videos:
        if os.path.exists(video_path):
            print(f"✓ Found: {video_path}")
        else:
            print(f"✗ Missing: {video_path}")
    
    # Ask for correct paths if files not found
    if not all(os.path.exists(path) for path in videos):
        print("\nPlease enter the correct paths to your video files:")
        real_video = input("Path to real video (e.g., /path/to/real.mp4): ")
        fake_video = input("Path to fake video (e.g., /path/to/fake.mp4): ")
        videos = [real_video, fake_video]
    
    # Test videos
    for video_path in videos:
        test_video(video_path) 