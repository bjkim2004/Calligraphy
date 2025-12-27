import { useState, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [text, setText] = useState('Hello World')
  const [fontType, setFontType] = useState('gaegu')
  const [artisticLevel, setArtisticLevel] = useState('high')
  const [brushIntensity, setBrushIntensity] = useState('medium')
  const [useAI, setUseAI] = useState(false)
  const [aiStrength, setAiStrength] = useState(0.38)
  const [isLoading, setIsLoading] = useState(false)
  const [imageUrl, setImageUrl] = useState(null)
  const [error, setError] = useState(null)
  const [seed, setSeed] = useState(null)
  const [fonts, setFonts] = useState([])
  const [categories, setCategories] = useState({})
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [aiStatus, setAiStatus] = useState({
    available: false,
    model_loaded: false,
    device: null,
    is_gpu: false
  })
  const [isLoadingAI, setIsLoadingAI] = useState(false)

  useEffect(() => {
    const loadData = async () => {
      try {
        const fontsRes = await fetch(`${API_URL}/fonts`)
        if (fontsRes.ok) {
          const data = await fontsRes.json()
          setFonts(data.fonts || [])
          setCategories(data.categories || {})
          if (data.fonts && data.fonts.length > 0) {
            setFontType(data.fonts[0].id)
          }
        }
        
        const aiRes = await fetch(`${API_URL}/ai/status`)
        if (aiRes.ok) {
          const aiData = await aiRes.json()
          setAiStatus(aiData)
        }
      } catch (err) {
        console.error('Data load failed:', err)
      }
    }
    loadData()
  }, [])

  const loadAIModel = useCallback(async () => {
    setIsLoadingAI(true)
    try {
      const response = await fetch(`${API_URL}/ai/load`, { method: 'POST' })
      if (response.ok) {
        const data = await response.json()
        setAiStatus(prev => ({ ...prev, model_loaded: true }))
        console.log('AI model loaded:', data)
      }
    } catch (err) {
      console.error('AI model load failed:', err)
    } finally {
      setIsLoadingAI(false)
    }
  }, [])

  const generateCalligraphy = useCallback(async () => {
    if (!text.trim()) {
      setError('Please enter text')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          font_type: fontType,
          artistic_level: artisticLevel,
          brush_intensity: brushIntensity,
          use_ai: useAI && aiStatus.available,
          ai_strength: aiStrength,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Generation failed')
      }

      const seedFromHeader = response.headers.get('X-Seed')
      setSeed(seedFromHeader)

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      
      if (imageUrl) URL.revokeObjectURL(imageUrl)
      setImageUrl(url)
    } catch (err) {
      setError(err.message || 'Server connection failed')
    } finally {
      setIsLoading(false)
    }
  }, [text, fontType, artisticLevel, brushIntensity, useAI, aiStrength, aiStatus.available, imageUrl])

  const downloadImage = useCallback(() => {
    if (!imageUrl) return
    const link = document.createElement('a')
    link.href = imageUrl
    link.download = `calligraphy_${seed || Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [imageUrl, seed])

  const filteredFonts = selectedCategory === 'all' 
    ? fonts 
    : fonts.filter(f => f.category === selectedCategory)

  const categoryIcons = {
    brush: 'B',
    handwriting: 'H',
    cute: 'C',
    modern: 'M',
    title: 'T'
  }

  const categoryNames = {
    brush: 'Brush',
    handwriting: 'Handwriting',
    cute: 'Cute',
    modern: 'Modern',
    title: 'Title'
  }

  const getAIBadge = () => {
    if (!aiStatus.available) {
      return { text: 'N/A', className: 'disabled' }
    }
    if (aiStatus.is_gpu) {
      return { text: 'GPU', className: 'gpu' }
    }
    return { text: 'CPU', className: 'cpu' }
  }

  const aiBadge = getAIBadge()

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="title-brush">C</span>
            <span className="title-main">Calligraphy Studio</span>
          </h1>
          <p className="subtitle">Create beautiful Korean calligraphy with various handwriting fonts</p>
        </header>

        <main className="main">
          <section className="input-section">
            <div className="input-group">
              <label className="label">Text Input</label>
              <input
                type="text"
                className="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter Korean text..."
                maxLength={50}
              />
              <span className="char-count">{text.length}/50</span>
            </div>

            <div className="option-group font-category-group">
              <label className="label">Category</label>
              <div className="category-tabs">
                <button
                  className={`category-tab ${selectedCategory === 'all' ? 'active' : ''}`}
                  onClick={() => setSelectedCategory('all')}
                >
                  <span>All</span>
                </button>
                {Object.keys(categories).map((catId) => (
                  <button
                    key={catId}
                    className={`category-tab ${selectedCategory === catId ? 'active' : ''}`}
                    onClick={() => setSelectedCategory(catId)}
                  >
                    <span>{categoryNames[catId] || catId}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="option-group">
              <label className="label">
                Font Style
                <span className="font-count">({filteredFonts.length})</span>
              </label>
              <div className="font-grid">
                {filteredFonts.map((font) => (
                  <button
                    key={font.id}
                    className={`font-btn ${fontType === font.id ? 'active' : ''}`}
                    onClick={() => setFontType(font.id)}
                  >
                    <span className="font-icon">{categoryIcons[font.category] || 'F'}</span>
                    <span className="font-name">{font.name}</span>
                    <span className="font-style">{font.style}</span>
                  </button>
                ))}
                {filteredFonts.length === 0 && (
                  <div className="no-fonts">
                    Please download fonts first.<br/>
                    <code>python download_fonts.py</code>
                  </div>
                )}
              </div>
            </div>

            <div className="options-row">
              <div className="option-group half">
                <label className="label">Artistic Level</label>
                <div className="radio-group">
                  {[
                    { value: 'low', label: 'Clean' },
                    { value: 'medium', label: 'Natural' },
                    { value: 'high', label: 'Artistic' },
                  ].map((option) => (
                    <button
                      key={option.value}
                      className={`radio-btn ${artisticLevel === option.value ? 'active' : ''}`}
                      onClick={() => setArtisticLevel(option.value)}
                    >
                      <span className="radio-text">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="option-group half">
                <label className="label">Brush Intensity</label>
                <div className="radio-group">
                  {[
                    { value: 'light', label: 'Light' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'strong', label: 'Strong' },
                  ].map((option) => (
                    <button
                      key={option.value}
                      className={`radio-btn ${brushIntensity === option.value ? 'active' : ''}`}
                      onClick={() => setBrushIntensity(option.value)}
                    >
                      <span className="radio-text">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="option-group ai-option-group">
              <div className="ai-header">
                <label className="label">
                  <span className="ai-icon">AI</span>
                  AI Texture Enhancement
                  <span className={`ai-badge ${aiBadge.className}`}>{aiBadge.text}</span>
                </label>
              </div>
              
              <div className="ai-content">
                <div className="ai-toggle-row">
                  <button
                    className={`ai-toggle ${useAI ? 'active' : ''}`}
                    onClick={() => setUseAI(!useAI)}
                    disabled={!aiStatus.available}
                  >
                    <span className="toggle-track">
                      <span className="toggle-thumb"></span>
                    </span>
                    <span className="toggle-label">
                      {useAI ? 'AI Enabled' : 'AI Disabled'}
                    </span>
                  </button>
                  
                  {useAI && aiStatus.available && (
                    <div className="ai-strength-control">
                      <label className="strength-label">Strength: {aiStrength.toFixed(2)}</label>
                      <input
                        type="range"
                        min="0.25"
                        max="0.50"
                        step="0.01"
                        value={aiStrength}
                        onChange={(e) => setAiStrength(parseFloat(e.target.value))}
                        className="strength-slider"
                      />
                      <div className="strength-hints">
                        <span>Low</span>
                        <span className="recommended">Recommended: 0.35-0.40</span>
                        <span>High</span>
                      </div>
                    </div>
                  )}
                </div>
                
                {useAI && aiStatus.available && !aiStatus.model_loaded && (
                  <div className="ai-load-section">
                    <button 
                      className={`ai-load-btn ${isLoadingAI ? 'loading' : ''}`}
                      onClick={loadAIModel}
                      disabled={isLoadingAI}
                    >
                      {isLoadingAI ? (
                        <>
                          <span className="spinner-small"></span>
                          <span>Loading model...</span>
                        </>
                      ) : (
                        <>
                          <span>Load AI Model</span>
                        </>
                      )}
                    </button>
                    <span className="ai-load-hint">First use requires model download</span>
                  </div>
                )}
                
                <p className="ai-description">
                  {!aiStatus.available ? (
                    'PyTorch/Diffusers not installed'
                  ) : useAI ? (
                    aiStatus.is_gpu ? (
                      'AI adds natural paper texture and ink effects (GPU, 10-30s)'
                    ) : (
                      'CPU mode: Generation may take 1-5 minutes'
                    )
                  ) : (
                    'Using fonts and OpenCV effects only (100% readable)'
                  )}
                </p>
              </div>
            </div>

            <button
              className={`generate-btn ${isLoading ? 'loading' : ''}`}
              onClick={generateCalligraphy}
              disabled={isLoading || !text.trim()}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span>
                  <span>{useAI && aiStatus.available && !aiStatus.is_gpu ? 'Processing (1-5min)...' : 'Generating...'}</span>
                </>
              ) : (
                <>
                  <span className="btn-icon">+</span>
                  <span>Generate Calligraphy</span>
                </>
              )}
            </button>

            {error && (
              <div className="error-message">
                <span>!</span>
                {error}
              </div>
            )}
          </section>

          <section className="result-section">
            {imageUrl ? (
              <div className="result-container">
                <div className="result-frame">
                  <img
                    src={imageUrl}
                    alt="Generated calligraphy"
                    className="result-image"
                  />
                </div>
                <div className="result-info">
                  <span className="result-font">
                    {fonts.find(f => f.id === fontType)?.name || fontType}
                  </span>
                  {useAI && aiStatus.available && (
                    <span className="result-ai">
                      AI Applied ({aiStatus.is_gpu ? 'GPU' : 'CPU'})
                    </span>
                  )}
                </div>
                <div className="result-actions">
                  <button className="action-btn download" onClick={downloadImage}>
                    <span>Download</span>
                  </button>
                  <button className="action-btn regenerate" onClick={generateCalligraphy}>
                    <span>Regenerate</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="placeholder">
                <div className="placeholder-icon">+</div>
                <p className="placeholder-text">
                  Enter text and<br />click Generate
                </p>
              </div>
            )}
          </section>
        </main>

        <footer className="footer">
          <p>Korean Calligraphy Generator - AI Enhanced</p>
        </footer>
      </div>
    </div>
  )
}

export default App
