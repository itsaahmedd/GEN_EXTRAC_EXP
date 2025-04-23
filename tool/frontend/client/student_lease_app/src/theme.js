import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#6A1B9A',  // Manchester Purple
    },
    secondary: {
      main: '#FFB300',  // A golden accent
    },
    error: {
      main: '#d32f2f',
    },
    // etc.
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    // ...
  },
});

export default theme;
