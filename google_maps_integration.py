"""
Google Maps API Integration for Road Condition Reporting
"""

import googlemaps
import json
import os
from datetime import datetime
from config import (
    GOOGLE_MAPS_API_KEY, 
    USE_MOCK_GPS, 
    MOCK_GPS_LOCATION,
    REPORTS_FILE,
    LOGS_DIR
)


class GoogleMapsIntegration:
    """Handle Google Maps API operations"""
    
    def __init__(self, api_key=GOOGLE_MAPS_API_KEY):
        self.api_key = api_key
        self.gmaps = None
        self.use_mock_gps = USE_MOCK_GPS
        
        if api_key and api_key != "YOUR_GOOGLE_MAPS_API_KEY_HERE":
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                print("✓ Google Maps API connected")
            except Exception as e:
                print(f"⚠ Google Maps API error: {e}")
        else:
            print("⚠ Google Maps API key not configured")
            print("  Set GOOGLE_MAPS_API_KEY in config.py")
    
    def get_current_location(self):
        """Get current GPS location"""
        if self.use_mock_gps:
            print(f"Using mock GPS: {MOCK_GPS_LOCATION}")
            return MOCK_GPS_LOCATION
        
        # TODO: Implement real GPS module integration
        # For now, return mock location
        return MOCK_GPS_LOCATION
    
    def reverse_geocode(self, latitude, longitude):
        """Convert coordinates to address"""
        if not self.gmaps:
            return f"{latitude}, {longitude}"
        
        try:
            result = self.gmaps.reverse_geocode((latitude, longitude))
            if result:
                return result[0]['formatted_address']
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return f"{latitude}, {longitude}"
    
    def report_road_condition(self, condition, confidence, latitude, longitude):
        """Report road condition to Google Maps and local log"""
        timestamp = datetime.now().isoformat()
        address = self.reverse_geocode(latitude, longitude)
        
        report = {
            'timestamp': timestamp,
            'condition': condition,
            'confidence': confidence,
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'address': address
            }
        }
        
        # Save to local log
        self._save_report(report)
        
        # TODO: Submit to Google Maps API when available
        print(f"\n=== Road Condition Report ===")
        print(f"Condition: {condition}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Location: {address}")
        print(f"Coordinates: ({latitude}, {longitude})")
        
        return report
    
    def _save_report(self, report):
        """Save report to JSON file"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        reports = []
        if os.path.exists(REPORTS_FILE):
            try:
                with open(REPORTS_FILE, 'r') as f:
                    reports = json.load(f)
            except:
                reports = []
        
        reports.append(report)
        
        with open(REPORTS_FILE, 'w') as f:
            json.dump(reports, f, indent=2)
        
        print(f"✓ Report saved to {REPORTS_FILE}")
    
    def get_nearby_roads(self, latitude, longitude, radius=1000):
        """Get nearby roads (for future enhancement)"""
        if not self.gmaps:
            return []
        
        try:
            places_result = self.gmaps.places_nearby(
                location=(latitude, longitude),
                radius=radius,
                type='route'
            )
            return places_result.get('results', [])
        except Exception as e:
            print(f"Error fetching nearby roads: {e}")
            return []
    
    def create_issue_report(self, condition, latitude, longitude, image_path=None):
        """Create detailed issue report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'issue_type': condition,
            'severity': self._get_severity(condition),
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'address': self.reverse_geocode(latitude, longitude)
            },
            'status': 'reported',
            'image': image_path
        }
        
        # Save detailed report
        detailed_file = os.path.join(LOGS_DIR, 'detailed_reports.json')
        detailed_reports = []
        
        if os.path.exists(detailed_file):
            try:
                with open(detailed_file, 'r') as f:
                    detailed_reports = json.load(f)
            except:
                detailed_reports = []
        
        detailed_reports.append(report)
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_reports, f, indent=2)
        
        return report
    
    def _get_severity(self, condition):
        """Determine severity level"""
        severity_map = {
            'Good': 'none',
            'Minor_Damage': 'low',
            'Crack': 'medium',
            'Pothole': 'high',
            'Severe_Damage': 'critical'
        }
        return severity_map.get(condition, 'unknown')


if __name__ == "__main__":
    # Test integration
    maps = GoogleMapsIntegration()
    
    location = maps.get_current_location()
    print(f"\nCurrent location: {location}")
    
    address = maps.reverse_geocode(location['latitude'], location['longitude'])
    print(f"Address: {address}")
    
    report = maps.report_road_condition(
        condition="Pothole",
        confidence=0.92,
        latitude=location['latitude'],
        longitude=location['longitude']
    )
    
    print("\n✓ Google Maps integration test successful!")
