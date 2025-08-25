# Warringu App - Specialised Shared Services Hub (Web Application)

A Flutter web application providing integrated access to specialist services for the Warringu organization.

## 🌐 Web Application Features

### Landing Page
- **Responsive two-column layout** (automatically adapts to mobile and desktop)
- **Left Column**: Embedded Koori Grapevine website (https://koorigrapevine.org.au/)
- **Right Column**: Access to Specialist Services Portal
- **Progressive Web App (PWA)** support for offline functionality

### Specialist Services Portal
- **Secure user authentication** with three user types:
  - Team Leaders / Organizational Managers
  - Client-Facing Case Managers
  - Residents (Current, Potential & Transitioned)

### Integrated Services
- **Wominjeka**: Case Management System
- **HR & Payroll**: Staff Management
- **P2i System**: Program Integration
- **SHiPP**: Support Services

### Database Integration
- **Local Storage**: Browser-based storage for session data
- **Cloud Integration**: API connections to Firebase and existing systems
- **Legacy Systems**: RESTful API integration with existing databases

### Training & Compliance Modules
- GEMS Training & Measures
- Circles of Security (CoC)
- Safe and Together
- Narrative Practice

## 🚀 Quick Start (Web Deployment)

### Prerequisites
- Flutter SDK (>=3.10.0) with web support enabled
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Web server for hosting (Apache, Nginx, or hosting service)

### One-Click Deployment

**For Linux/macOS:**
```bash
chmod +x deploy-web.sh
./deploy-web.sh
```

**For Windows:**
```powershell
.\deploy-web.ps1
```

### Manual Build Process

1. **Navigate to project directory**:
   ```bash
   cd /path/to/warringu-app
   ```

2. **Install dependencies**:
   ```bash
   flutter pub get
   ```

3. **Build for web**:
   ```bash
   flutter build web --web-renderer html --release
   ```

4. **Deploy to web server**:
   ```bash
   # Copy contents of build/web/ to your web server
   cp -r build/web/* /path/to/your/webserver/
   ```

### Local Testing

1. **Build the app**:
   ```bash
   flutter build web
   ```

2. **Serve locally**:
   ```bash
   cd build/web
   python -m http.server 8000
   ```

3. **Open in browser**: http://localhost:8000

## 📁 Project Structure

```
warringu-app/
├── lib/
│   ├── main.dart                     # Web-optimized app entry point
│   ├── screens/
│   │   ├── landing_page.dart         # Responsive landing page with iframe
│   │   └── specialist_portal.dart     # Specialist services portal
│   └── widgets/
│       ├── user_type_selector.dart           # User authentication widget
│       └── service_integration_panel.dart    # Service integration display
├── web/                              # Web-specific files
│   ├── index.html                    # Main HTML template
│   ├── manifest.json                 # PWA configuration
│   └── icons/                        # Web app icons
├── assets/
│   └── images/                       # App images and assets
├── deploy-web.sh                     # Linux/macOS deployment script
├── deploy-web.ps1                    # Windows deployment script
└── pubspec.yaml                      # Web-optimized dependencies
```

## 🔧 Web-Specific Configuration

### iframe Integration
- **Koori Grapevine**: Embedded using secure iframe with CORS handling
- **Cross-origin support**: Configured for external website embedding
- **Responsive design**: Iframe scales with container

### Progressive Web App (PWA)
- **Offline support**: Service worker for caching
- **Add to home screen**: Mobile device installation
- **App manifest**: Native app-like experience

### Browser Compatibility
- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support with minor limitations
- **Edge**: Full support
- **Mobile browsers**: Responsive design optimized

## 🛡️ Security Features

### Web Security
- **Content Security Policy**: Configured for iframe embedding
- **HTTPS required**: Secure transmission for production
- **Session management**: Browser-based secure storage
- **CORS handling**: Proper cross-origin resource sharing

### User Authentication
- **Role-based access control**
- **Secure session handling**
- **Activity logging**
- **Multi-factor authentication ready**

## 🎯 User Types & Web Access

### 1. Team Leaders / Organizational Managers
- **Full dashboard access** via web browser
- **Administrative controls** optimized for desktop
- **Reporting tools** with export capabilities
- **System management** through web interface

### 2. Client-Facing Case Managers
- **Case management tools** accessible on any device
- **Mobile-responsive** interface for field work
- **Quick access** to client information
- **Offline capability** for essential functions

### 3. Residents
- **Simplified interface** for easy navigation
- **Mobile-friendly** design for smartphone access
- **Resource access** through web portal
- **Communication tools** integrated

## 🌍 Deployment Options

### 1. Static Web Hosting
- **GitHub Pages**: Free hosting for public repositories
- **Netlify**: Drag-and-drop deployment with CI/CD
- **Vercel**: Automatic deployments from Git
- **Firebase Hosting**: Google's web hosting solution

### 2. Traditional Web Servers
- **Apache**: Standard web server deployment
- **Nginx**: High-performance web server
- **IIS**: Windows-based web server
- **Cloud providers**: AWS, Azure, Google Cloud

### 3. Content Delivery Network (CDN)
- **Global distribution** for faster loading
- **Caching strategy** for better performance
- **SSL termination** for security

## 📊 Performance Optimizations

### Web-Specific Optimizations
- **HTML renderer**: Optimized for web browsers
- **Code splitting**: Lazy loading for better performance
- **Asset optimization**: Compressed images and resources
- **Caching strategy**: Browser and service worker caching

### Responsive Design
- **Mobile-first approach**: Optimized for smallest screens first
- **Breakpoint management**: Smooth transitions between device sizes
- **Touch-friendly interface**: Appropriate touch targets
- **Fast loading**: Optimized for various connection speeds

## 🔗 Integration Capabilities

### API Connections
- **RESTful APIs**: Standard HTTP API integration
- **WebSocket support**: Real-time data updates
- **OAuth integration**: Third-party authentication
- **CORS configuration**: Cross-origin resource sharing

### External Services
- **Wominjeka API**: Case management system integration
- **HR systems**: Staff and payroll management
- **P2i integration**: Program data synchronization
- **SHiPP services**: Support service connections

## 📱 Mobile Web Features

### Progressive Web App
- **Add to home screen**: Native app-like installation
- **Offline functionality**: Essential features work offline
- **Push notifications**: Web-based notification system
- **Background sync**: Data synchronization when online

### Mobile Optimization
- **Touch gestures**: Swipe, tap, and pinch support
- **Viewport optimization**: Proper scaling on all devices
- **Performance**: Optimized for mobile networks
- **Battery efficiency**: Minimal resource usage

## 🛠️ Development

### Hot Reload for Web
```bash
flutter run -d chrome --web-port=8080
```

### Debug Mode
```bash
flutter run -d web-server --web-port=8080 --debug
```

### Production Build
```bash
flutter build web --release --web-renderer html
```

## 📈 Analytics & Monitoring

### Web Analytics Integration Ready
- **Google Analytics**: User behavior tracking
- **Performance monitoring**: Core web vitals
- **Error tracking**: Crash and error reporting
- **User engagement**: Session and interaction metrics

## 🆘 Troubleshooting

### Common Issues

**1. iframe not loading:**
- Check CORS policy on target website
- Verify internet connection
- Ensure HTTPS in production

**2. Slow loading:**
- Check network connection
- Clear browser cache
- Verify CDN configuration

**3. Mobile display issues:**
- Test on actual devices
- Check viewport meta tags
- Verify responsive breakpoints

### Support Resources
- **Flutter Web Documentation**: https://flutter.dev/web
- **Deployment Guides**: Available for all major platforms
- **Community Support**: Flutter web community forums

## 📄 License

This project is proprietary software developed for Warringu organization.

---

**🌐 Ready for web deployment!** Use the deployment scripts for quick setup or follow the manual process for custom configurations.
