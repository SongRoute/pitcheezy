// Club colours reused from codex/mobile-atbat-1005 b73fa10 (WatchAlong.tsx TEAM_COLORS), accepted in review 5438a68.
export const TEAM_COLORS: Record<string, string> = {
  'Arizona Diamondbacks': '#A71930', 'Athletics': '#003831', 'Atlanta Braves': '#CE1141', 'Baltimore Orioles': '#DF4601',
  'Boston Red Sox': '#BD3039', 'Chicago Cubs': '#0E3386', 'Chicago White Sox': '#27251F', 'Cincinnati Reds': '#C6011F',
  'Cleveland Guardians': '#00385D', 'Colorado Rockies': '#333366', 'Detroit Tigers': '#0C2340', 'Houston Astros': '#EB6E1F',
  'Kansas City Royals': '#004687', 'Los Angeles Angels': '#BA0021', 'Los Angeles Dodgers': '#005A9C', 'Miami Marlins': '#00A3E0',
  'Milwaukee Brewers': '#12284B', 'Minnesota Twins': '#002B5C', 'New York Mets': '#002D72', 'New York Yankees': '#0C2340',
  'Philadelphia Phillies': '#E81828', 'Pittsburgh Pirates': '#FDB827', 'San Diego Padres': '#2F241D',
  'San Francisco Giants': '#FD5A1E', 'Seattle Mariners': '#0C2C56', 'St. Louis Cardinals': '#C41E3A',
  'Tampa Bay Rays': '#092C5C', 'Texas Rangers': '#003278', 'Toronto Blue Jays': '#134A8E', 'Washington Nationals': '#AB0003' };
export const teamColor = (team: string) => TEAM_COLORS[team] ?? '#8a948f';  // unknown team: neutral grey
