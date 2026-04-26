# Header Component & React Router Setup Guide

## Installation

First, install React Router DOM:

```bash
npm install react-router-dom
```

Update your `package.json` dependencies with:

```bash
npm install
```

## Implementation Summary

### New File Structure

1. **`src/components/Header.tsx`** ✅ (Created)
   - Shared header component used by all pages
   - Accepts `type` prop: `'main'` for landing page, `'onboarding'` for onboarding page
   - Handles navigation and active link highlights

2. **`src/App.tsx`** ✅ (Updated)
   - Now uses React Router for page routing
   - Two routes:
     - `/` → LandingScreen (main page)
     - `/onboarding` → OnboardingPage

3. **`src/screens/LandingScreen.tsx`** ✅ (Updated)
   - Removed navbar code (moved to Header component)
   - Uses `useNavigate()` hook for navigation
   - Imports and uses `<Header type="main" />`
   - "Essayer gratuitement" button navigates to `/onboarding`

4. **`src/screens/OnboardingPage.tsx`** ✅ (Updated)
   - Removed navbar code (moved to Header component)
   - Imports and uses `<Header type="onboarding" />`
   - Logo click navigates back to `/` (home page)

## How It Works

### Main Page Header
- Light background with navigation bar
- Shows: Logo | Navigation Links (Platform, Services, Pricing, About) | Get Started Button
- Clicking "Platform" scrolls to top
- Active link highlighted in purple with underline

### Onboarding Page Header
- Light background with 4 utility icons
- Shows: Logo | Icons (upload, bookmark, history, edit) | Sign In & Register buttons
- Clicking logo navigates back to main page `/`

### Navigation Flow

```
Home Page (/)
    ↓ (Click "Essayer gratuitement")
    ↓
Onboarding Page (/onboarding)
    ↓ (Click logo or back button)
    ↓
Home Page (/)
```

## Code Usage

### Import Header Component

```tsx
import Header from '@/components/Header'

export default function MyPage() {
  const [activeLink, setActiveLink] = useState('platform')

  return (
    <>
      <Header 
        type="main" 
        activeLink={activeLink} 
        onLinkChange={setActiveLink} 
      />
      {/* Your page content */}
    </>
  )
}
```

### Using React Router Navigation

```tsx
import { useNavigate } from 'react-router-dom'

export default function MyComponent() {
  const navigate = useNavigate()

  return (
    <button onClick={() => navigate('/onboarding')}>
      Go to Onboarding
    </button>
  )
}
```

## Header Component Props

```typescript
interface HeaderProps {
  type: 'main' | 'onboarding'           // Type of header to display
  activeLink?: string                    // Current active link ID
  onLinkChange?: (linkId: string) => void // Callback when link changes
}
```

## Features

✅ Two distinct header designs based on page type
✅ Smooth scroll navigation with active state highlighting
✅ React Router integration for proper page navigation
✅ Logo click behavior changes based on current page:
   - Main page: scrolls to top
   - Onboarding page: navigates back to home
✅ Consistent UI/UX across all pages
✅ Responsive design (mobile and desktop)

## Next Steps

1. Run your app to test the new routing:
   ```bash
   npm run dev
   ```

2. Test the navigation:
   - Click "Essayer gratuitement" to go to onboarding page
   - Click logo on onboarding page to return to home
   - Scroll on main page to see active link highlighting
   - Click nav links to scroll to sections

## Troubleshooting

**Issue**: `Cannot find module 'react-router-dom'`
- Solution: Run `npm install react-router-dom`

**Issue**: Navigation not working
- Check that you're using `useNavigate()` hook from react-router-dom
- Ensure app structure wraps everything in `<Router>`

**Issue**: Header not displaying
- Verify Header is imported: `import Header from '@/components/Header'`
- Check that type prop is either `'main'` or `'onboarding'`
