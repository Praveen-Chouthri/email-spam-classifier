# Vercel Deployment Guide

## Prerequisites
1. GitHub repository with your code
2. Vercel account (free tier available)
3. Vercel CLI installed (optional but recommended)

## Step-by-Step Deployment

### 1. Install Vercel CLI (Optional)
```bash
npm i -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy via Vercel Dashboard (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Configure the following settings:
   - **Framework Preset**: Other
   - **Root Directory**: ./
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements-vercel.txt`

### 4. Set Environment Variables

In your Vercel project dashboard, go to Settings > Environment Variables and add:

```
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key-here
APP_NAME=Email Spam Classifier
DEFAULT_MODEL=naive_bayes
MAX_BATCH_SIZE=100
CLASSIFICATION_TIMEOUT=10.0
LOG_LEVEL=INFO
FORCE_HTTPS=True
SESSION_COOKIE_SECURE=True
```

### 5. Deploy via CLI (Alternative)
```bash
vercel --prod
```

## Important Notes

### Limitations on Vercel
1. **File System**: Vercel functions are stateless. Uploaded files and models won't persist between requests
2. **Cold Starts**: First request might be slower due to function initialization
3. **Memory Limits**: Free tier has 1GB memory limit
4. **Execution Time**: 30-second timeout for serverless functions

### Recommended Modifications for Production

1. **Use External Storage**: Store models and data in cloud storage (AWS S3, Google Cloud Storage)
2. **Database Integration**: Use a cloud database for persistent data
3. **Model Loading**: Implement lazy loading or model caching strategies
4. **File Uploads**: Use cloud storage for file uploads instead of local filesystem

### Testing Your Deployment

1. Visit your Vercel URL
2. Test the main routes:
   - `/` - Home page
   - `/api/health` - Health check
   - `/classify` - Classification interface

### Troubleshooting

1. **Build Errors**: Check the build logs in Vercel dashboard
2. **Runtime Errors**: Check function logs in Vercel dashboard
3. **Import Errors**: Ensure all dependencies are in requirements-vercel.txt
4. **Timeout Issues**: Optimize model loading and processing

## Next Steps

1. Set up monitoring and logging
2. Implement proper error handling for serverless environment
3. Consider upgrading to Vercel Pro for better performance
4. Set up CI/CD pipeline for automatic deployments