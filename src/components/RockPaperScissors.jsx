import { useState, useEffect, useRef } from 'react';

const RockPaperScissors = () => {
  const [playerScore, setPlayerScore] = useState(0);
  const [computerScore, setComputerScore] = useState(0);
  const [result, setResult] = useState('');
  const [gesture, setGesture] = useState('Waiting...');
  const [gameHistory, setGameHistory] = useState([]);
  const [streak, setStreak] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [isCameraConnected, setIsCameraConnected] = useState(false);
  const [isPythonConnected, setIsPythonConnected] = useState(false);
  const [computerChoice, setComputerChoice] = useState(null);
  const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });
  const [scrollPosition, setScrollPosition] = useState(0);
  const ws = useRef(null);
  const recentMatchesRef = useRef(null);

  const isPointInRect = (x, y, rect) => {
    return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
  };

  useEffect(() => {
    const checkCamera = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const hasCamera = devices.some(device => device.kind === 'videoinput');
        setIsCameraConnected(hasCamera);
      } catch (err) {
        console.error('Camera error:', err);
        setIsCameraConnected(false);
      }
    };

    ws.current = new WebSocket('ws://localhost:8765');
    
    ws.current.onopen = () => setIsPythonConnected(true);
    ws.current.onclose = () => setIsPythonConnected(false);
    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received:', data);

        if (data.type === 'handPosition') {
          const cursorX = data.x * window.innerWidth;
          const cursorY = data.y * window.innerHeight;
          setCursorPosition({ x: cursorX, y: cursorY });

          // Buton kontrolü için cursor pozisyonunu kontrol et
          const startButton = document.querySelector('.start-button');
          const resetButton = document.querySelector('.reset-button');
          const recentMatchesArea = recentMatchesRef.current;
          
          if (startButton && resetButton) {
            const startRect = startButton.getBoundingClientRect();
            const resetRect = resetButton.getBoundingClientRect();
            
            // Cursor butonların üzerinde mi ve tıklama var mı kontrol et
            if (data.isClicking) {
              if (isPointInRect(cursorX, cursorY, startRect)) {
                startGame();
              } else if (isPointInRect(cursorX, cursorY, resetRect)) {
                resetGame();
              }
            }
          }

          // Scroll kontrolü
          if (data.scrollDirection && recentMatchesRef.current) {
            const recentMatchesRect = recentMatchesRef.current.getBoundingClientRect();
            
            // Cursor Recent Matches alanının üzerinde mi kontrol et
            if (isPointInRect(cursorX, cursorY, recentMatchesRect)) {
              const scrollAmount = 30;
              if (data.scrollDirection === 'up') {
                recentMatchesRef.current.scrollTop -= scrollAmount;
              } else if (data.scrollDirection === 'down') {
                recentMatchesRef.current.scrollTop += scrollAmount;
              }
            }
          }
        } else if (data.type === 'gestureUpdate' && data.gesture) {
          setGesture(data.gesture);
        } else if (data.type === 'gameResult') {
          setResult(data.result);
          setComputerChoice(data.computerMove);
          updateGame(data);
        } else if (data.status === 'connected') {
          console.log('Connected to Python server');
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    checkCamera();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const updateGame = (data) => {
    if (data.result === 'Win') {
      setPlayerScore(prev => prev + 1);
      setStreak(prev => prev + 1);
    } else if (data.result === 'Lose') {
      setComputerScore(prev => prev + 1);
      setStreak(0);
    }
    setGameHistory(prev => [{
      playerMove: data.playerMove,
      computerMove: data.computerMove,
      result: data.result
    }, ...prev].slice(0, 10));
  };

  const startGame = () => {
    setIsPlaying(true);
    setResult('');
    setComputerChoice(null);
    let count = 3;
    const timer = setInterval(() => {
      if (count > 0) {
        setCountdown(count - 1);
        count--;
      } else {
        clearInterval(timer);
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'gameStart' }));
        }
        setCountdown(3);
        setIsPlaying(false);
      }
    }, 1000);
  };

  const resetGame = () => {
    setPlayerScore(0);
    setComputerScore(0);
    setStreak(0);
    setGameHistory([]);
    setResult('');
    setGesture('Waiting...');
    setComputerChoice(null);
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'reset' }));
    }
  };

  const gestures = {
    rock: '✊',
    paper: '✋',
    scissors: '✌️'
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 text-white p-4 relative">
      {/* Cursor */}
      <div 
        className="fixed pointer-events-none z-50"
        style={{
          left: `${cursorPosition.x}px`,
          top: `${cursorPosition.y}px`,
          transform: 'translate(-50%, -50%)',
          transition: 'all 0.05s ease',
          width: '24px',
          height: '24px'
        }}
      >
        <div className="absolute inset-0 rounded-full bg-blue-500 opacity-30 animate-pulse"></div>
        <div className="absolute inset-2 rounded-full bg-blue-500 opacity-60"></div>
        <div className="absolute inset-3 rounded-full bg-white"></div>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="absolute top-4 right-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isCameraConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">Camera: {isCameraConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isPythonConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">Python: {isPythonConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-5xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
            Rock Paper Scissors
          </h1>
          <p className="text-lg text-blue-200">Gesture-Controlled Gaming</p>
        </div>

        <div className="flex justify-center gap-4 mb-8">
          <button 
            onClick={startGame}
            className="start-button px-6 py-3 bg-green-500 hover:bg-green-600 rounded-lg font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!isCameraConnected || !isPythonConnected || isPlaying}
          >
            {isPlaying ? `${countdown}...` : 'Start Game'}
          </button>
          <button 
            onClick={resetGame}
            className="reset-button px-6 py-3 bg-red-500 hover:bg-red-600 rounded-lg font-bold transition-all"
          >
            Reset
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur-sm">
            <div className="flex justify-between items-center mb-4">
              <div className="text-center">
                <h2 className="text-xl font-bold">You</h2>
                <p className="text-4xl font-bold text-blue-400">{playerScore}</p>
              </div>
              <div className="text-4xl font-bold">VS</div>
              <div className="text-center">
                <h2 className="text-xl font-bold">AI</h2>
                <p className="text-4xl font-bold text-purple-400">{computerScore}</p>
              </div>
            </div>
            <div className="text-center mt-4">
              <p className="text-sm text-blue-200">Current Streak</p>
              <p className="text-3xl font-bold text-yellow-400">{streak}</p>
            </div>
          </div>

          <div className="bg-white/10 rounded-xl p-6 backdrop-blur-sm">
            <div className="text-center">
              <h2 className="text-xl font-bold mb-4">Current Gesture</h2>
              <p className="text-6xl mb-4">{gestures[gesture.toLowerCase()] || '👋'}</p>
              <p className="text-lg bg-white/5 rounded-lg p-2">{gesture}</p>
              {!isPlaying && computerChoice && result && (
                <p className="text-xl mt-4">
                  Computer chose: {gestures[computerChoice.toLowerCase()]}
                </p>
              )}
              {result && (
                <p className={`text-xl mt-4 font-bold ${
                  result === 'Win' ? 'text-green-400' :
                  result === 'Lose' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {result}
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur-sm">
            <h2 className="text-xl font-bold mb-4 text-center">How to Play</h2>
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(gestures).map(([name, emoji]) => (
                <div key={name} className="text-center p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-all">
                  <p className="text-3xl mb-2">{emoji}</p>
                  <p className="capitalize">{name}</p>
                </div>
              ))}
            </div>
            <div className="text-sm text-center mt-4 text-blue-200 space-y-2">
              <p>Show your gesture to the camera when countdown ends</p>
              <p>Use index finger to hover and pinch to click buttons</p>
              <p>For Recent Matches: Hover and raise two fingers to scroll up/down</p>
            </div>
          </div>

          <div className="bg-white/10 rounded-xl p-6 backdrop-blur-sm">
            <h2 className="text-xl font-bold mb-4 text-center">Recent Matches</h2>
            <div ref={recentMatchesRef} className="max-h-48 overflow-y-auto">
              {gameHistory.map((game, index) => (
                <div 
                  key={index} 
                  className="flex justify-between items-center mb-2 p-2 bg-white/5 rounded-lg hover:bg-white/10 transition-all"
                >
                  <span className="text-2xl">{gestures[game.playerMove.toLowerCase()]}</span>
                  <span className={`text-sm font-bold ${
                    game.result === 'Win' ? 'text-green-400' :
                    game.result === 'Lose' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {game.result}
                  </span>
                  <span className="text-2xl">{gestures[game.computerMove.toLowerCase()]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RockPaperScissors;